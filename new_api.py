from dataclasses import dataclass
from collections.abc import Iterator
from typing import Callable, Optional
import weave
from weave import graph_client_context
import inspect
import itertools
import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from rich.table import Table
from weave.flow.chat_util import OpenAIStream

import query
import json
from pandas_util import find_rows_with_vals, get_unflat_value


def get_op_param_names(op: Callable):
    sig = inspect.signature(op)
    return list(sig.parameters.keys())


CACHE = {}


def config_hash(config):
    # Create a stable hash of a python object, including lists and dicts
    if isinstance(config, dict):
        return hash(tuple((k, config_hash(v)) for k, v in sorted(config.items())))
    if isinstance(config, list):
        return hash(tuple(config_hash(v) for v in config))
    try:
        return hash(config)
    except TypeError:
        return config_hash(config.__dict__)


@dataclass
class OpenSweep:
    op: weave.Op
    kwargs: dict

    @property
    def open_params(self):
        op_param_names = get_op_param_names(self.op)
        return [k for k in op_param_names if k not in self.kwargs]

    def param_names(self):
        return get_op_param_names(self.op)

    def __call__(self, **kwargs):
        op_param_names = get_op_param_names(self.op)
        for k in kwargs.keys():
            if k not in op_param_names:
                raise ValueError(f"Extra kwargs: {k}. Expected: {op_param_names}")
        kwargs = {**self.kwargs, **kwargs}
        open_sweep = OpenSweep(self.op, kwargs)
        if open_sweep.open_params:
            return open_sweep

        seqs = {k: v for k, v in kwargs.items() if isinstance(v, list)}
        singles = {k: v for k, v in kwargs.items() if not isinstance(v, list)}
        seq_names = seqs.keys()
        seqs = list(itertools.product(*seqs.values()))
        configs = [
            {**singles, **{name: seq[i] for i, name in enumerate(seq_names)}}
            for seq in seqs
        ]
        return ClosedSweep(self.op, configs)


@dataclass
class ClosedSweep:
    op: weave.Op
    configs: list

    def stats(self):
        return {
            "cache_hits": sum(config_hash(config) in CACHE for config in self.configs),
            "total": len(self.configs),
        }

    def compute(self, on_item_done=None):
        config_hashes = [config_hash(config) for config in self.configs]
        compute_configs = [
            config for config, h in zip(self.configs, config_hashes) if h not in CACHE
        ]
        cache_hits = len(self.configs) - len(compute_configs)
        print("cache_hits:", cache_hits)
        for i, config in tqdm.tqdm(enumerate(compute_configs)):
            h = config_hash(config)
            CACHE[h] = self.op(**config)
            if on_item_done:
                on_item_done((i + 1) / len(compute_configs), config, CACHE[h])
        return [(c, CACHE[h]) for c, h in zip(self.configs, config_hashes)]


def weave_sweep(op: weave.Op, **kwargs):
    return OpenSweep(op, {})(**kwargs)


def unique_items(l):
    unique_items = []
    seen = {}
    for item in l:
        if item not in seen:
            unique_items.append(item)
            seen[item] = True
    return list(seen.keys())


def weave_map(data, op: weave.Op):
    op_param_names = get_op_param_names(op)
    configs = []
    for _, d in data.iterrows():
        row = d.to_dict()
        conf = {}
        for k in op_param_names:
            if k not in row:
                raise ValueError(f"missing {k}")
            conf[k] = row[k]
        configs.append(conf)

    config_hashes = [config_hash(config) for config in configs]
    hash_configs = dict(zip(config_hashes, configs))
    # Hmm... notice unique items here...
    config_hashes = unique_items(config_hashes)
    configs = [hash_configs[h] for h in config_hashes]
    compute_configs = [
        config for config, h in zip(configs, config_hashes) if h not in CACHE
    ]
    cache_hits = len(configs) - len(compute_configs)
    print("cache_hits:", cache_hits)
    for config in tqdm.tqdm(compute_configs):
        h = config_hash(config)
        CACHE[h] = op(**config)
    return [(c, CACHE[h]) for c, h in zip(configs, config_hashes)]


@dataclass
class InputColumn:
    pass


@dataclass
class OpColumn:
    input_mapping: dict
    op: weave.Op


class NotComputed:
    def __repr__(self):
        return "<NotComputed>"


NOT_COMPUTED = NotComputed()


class OpResultCache:
    def __init__(self):
        self.results = {}

    def _get_one(self, op_ref, op_config):
        op_results = self.results.get(op_ref, {})
        input_hash = config_hash(op_config)
        if input_hash not in op_results:
            return NOT_COMPUTED
        return op_results

    def get_many(self, op_ref, op_configs):
        return [self._get_one(op_ref, oc) for oc in op_configs]

    def set(self, op_ref, op_config, result):
        if op_ref not in self.results:
            self.results[op_ref] = {}
        input_hash = config_hash(op_config)
        self.results[op_ref][input_hash] = result


class WeaveResultCache:
    def __init__(self):
        pass

    def get_many(self, op_ref, op_configs):
        wc = graph_client_context.require_graph_client()

        calls = query.get_calls(wc, op_ref)
        op_configs = [
            {k: v.ref.uri() if hasattr(v, "ref") else v for k, v in op_config.items()}
            for op_config in op_configs
        ]
        results = []
        for op_config in op_configs:
            find_config = pd.json_normalize([{"inputs": op_config}])
            rows = find_rows_with_vals(calls.df, find_config)

            if len(rows) == 0:
                results.append(NOT_COMPUTED)
            else:
                results.append(get_unflat_value(rows.iloc[0], "output"))
        return results

    def get_many(self, op_ref, op_configs):
        wc = graph_client_context.require_graph_client()
        calls = query.get_calls(wc, op_ref)

        # Convert op_configs to a DataFrame
        op_configs_df = pd.DataFrame(
            [
                {
                    k: v.ref.uri() if hasattr(v, "ref") else v
                    for k, v in op_config.items()
                }
                for op_config in op_configs
            ]
        )

        # Add an index column to keep track of the original order
        op_configs_df["original_index"] = range(len(op_configs_df))

        # Prepare the calls DataFrame
        calls_df = calls.df.astype(str)

        # Prepare the op_configs DataFrame
        op_configs_df_normalized = pd.json_normalize(
            op_configs_df.to_dict(orient="records"), sep="."
        )
        op_configs_df_normalized = op_configs_df_normalized.add_prefix("inputs.")
        op_configs_df_normalized = op_configs_df_normalized.astype(str)

        # Perform the merge operation
        merged_df = calls_df.merge(
            op_configs_df_normalized, how="right", indicator=True
        )

        # Group by the original index and aggregate the results
        grouped = merged_df.groupby("inputs.original_index")

        # Initialize results list with NOT_COMPUTED for all op_configs
        results = [NOT_COMPUTED] * len(op_configs)

        for index, group in grouped:
            if "both" in group["_merge"].values:
                # If there's at least one match, use the first match
                match_row = group[group["_merge"] == "both"].iloc[0]
                results[int(index)] = get_unflat_value(match_row, "output")

        return results

    def set(self, op_ref, op_config, result):
        pass


class ComputeTable:
    def __init__(self, initial_inputs: Optional[list] = None, trials: int = 1) -> None:
        self.result_cache = WeaveResultCache()
        self.columns = {}
        self.rows = []
        if initial_inputs:
            for k in initial_inputs[0]:
                self.add_input(k)
            if trials > 1:
                self.add_input("trial")
            for row in initial_inputs:
                if trials > 1:
                    for i in range(trials):
                        self.add_row({**row, "trial": i})
                else:
                    self.add_row(row)

    def add_input(self, name):
        self.columns[name] = InputColumn()

    def add_op(self, op: weave.Op, input_mapping: Optional[dict] = None):
        full_input_mapping = {}
        op_param_names = get_op_param_names(op)
        for param_name in op_param_names:
            if input_mapping and param_name in input_mapping:
                full_input_mapping[param_name] = input_mapping[param_name]
            else:
                full_input_mapping[param_name] = param_name
        for k, v in full_input_mapping.items():
            if v not in self.columns:
                raise ValueError(f"Missing input: {v}")
        self.columns[op.name] = OpColumn(full_input_mapping, op)
        for row in self.rows:
            row[op.name] = NOT_COMPUTED

    def add_row(self, row: dict):
        add_row = {}
        for col_name, col in self.columns.items():
            if isinstance(col, InputColumn):
                if col_name not in row:
                    raise ValueError(f"Missing input: {col_name}")
                add_row[col_name] = row[col_name]
            else:
                add_row[col_name] = NOT_COMPUTED
        self.rows.append(add_row)

    def get_op_configs(self, col_name):
        col = self.columns[col_name]
        if not isinstance(col, OpColumn):
            raise ValueError(f"Column {col_name} is not an OpColumn")
        return [
            {
                param_name: row[input_name]
                for param_name, input_name in col.input_mapping.items()
            }
            for row in self.rows
        ]

    def fill_from_cache(self):
        wc = graph_client_context.require_graph_client()
        for col_name, col in self.columns.items():
            if isinstance(col, OpColumn):
                ref = wc._save_op(col.op, col.op.name).uri()
                op_configs = self.get_op_configs(col_name)
                cached_results = self.result_cache.get_many(ref, op_configs)
                for row, result in zip(self.rows, cached_results):
                    row[col_name] = result

    def status(self):
        op_status = {}
        for col_name, col in self.columns.items():
            if isinstance(col, OpColumn):
                op_status[col_name] = {
                    "not_computed": sum(
                        1 for r in self.rows if r[col_name] is NOT_COMPUTED
                    ),
                }
        return op_status

    def execute(self):
        wc = graph_client_context.require_graph_client()
        for col_name, col in self.columns.items():
            work_to_do = {}
            if isinstance(col, OpColumn):
                ref = wc._save_op(col.op).uri()
                op_configs = self.get_op_configs(col_name)
                for i, (row, op_config) in enumerate(zip(self.rows, op_configs)):
                    value = row[col_name]
                    if value is not NOT_COMPUTED:
                        continue
                    work_to_do[(i, col_name)] = op_config

            with ThreadPoolExecutor(max_workers=16) as executor:

                def do_one(work):
                    with graph_client_context.set_graph_client(wc):
                        (i, col_name), op_config = work
                        col = self.columns[col_name]
                        try:
                            return col.op(**op_config)
                        except:
                            return None

                results = list(executor.map(do_one, work_to_do.items()))

            # for (i, col_name), op_config in work_to_do.items():
            #     row = self.rows[i]
            #     col = self.columns[col_name]
            #     print(i, col_name, op_config)
            #     result = col.op(**op_config)

            iterator_results = {}
            for ((i, col_name), op_config), result in zip(work_to_do.items(), results):
                row = self.rows[i]
                col = self.columns[col_name]
                # print(i, col_name, op_config)
                # result = col.op(**op_config)
                if isinstance(result, Iterator):
                    iterator_results[(i, col_name)] = {"iterator": result, "result": ""}
                else:
                    row[col_name] = result
                    yield {"row": i, "col": col_name, "val": result}
                    self.result_cache.set(col.op.ref.uri(), op_config, result)

            while iterator_results:
                stopped = []
                for (i, col_name), ir in iterator_results.items():
                    iterator = ir["iterator"]
                    try:
                        delta = next(iterator)
                        delta_content = delta.choices[0].delta.content
                        if delta_content:
                            ir["result"] += delta_content
                            self.rows[i][col_name] = ir["result"]
                            yield {"row": i, "col": col_name, "val": ir["result"]}
                    except StopIteration:
                        stopped.append((i, col_name))
                        self.result_cache.set(
                            self.columns[col_name].op.ref.uri(),
                            work_to_do[(i, col_name)],
                            ir["result"],
                        )
                for i, col_name in stopped:
                    del iterator_results[(i, col_name)]

                # for chunk in result:
                #     delta_content = chunk.choices[0].delta.content
                #     if delta_content:
                #         final_message += chunk.choices[0].delta.content
                #     row[col_name] = final_message
                #     try:
                #         yield {
                #             "row": i,
                #             "col": col_name,
                #             "val": final_message,
                #         }
                #     except ValueError:
                #         pass
                # result = final_message

    def dataframe(self):
        df = pd.DataFrame(self.rows, columns=self.columns.keys())
        return df

    def rich_table(self):
        table = Table()
        for col_name, col in self.columns.items():
            if isinstance(col, InputColumn):
                col_type = "i"
            elif isinstance(col, OpColumn):
                col_type = "o"
            else:
                raise ValueError(f"Unknown column type for column: {col_name}")
            table.add_column(f"{col_name} ({col_type})")
        for row in self.rows:
            table.add_row(
                *[
                    str(row[col_name]) if row[col_name] is not NOT_COMPUTED else "-"
                    for col_name in self.columns
                ]
            )
        return table
