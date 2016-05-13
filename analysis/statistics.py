#!/usr/bin/env python
# Copyright 2016 Alessio Sclocco <a.sclocco@vu.nl>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to retrieve statistical properties of the data."""

def get_quartiles(db_queue, table, benchmark, scenario):
    """Returns the quartiles performance, rounded to integers."""
    results = list()
    metrics = ""
    if benchmark.lower() == "triad":
        metrics = "GBs"
    elif benchmark.lower() == "reduction":
        metrics = "GBs"
    elif benchmark.lower() == "stencil":
        metrics = "GFLOPs"
    elif benchmark.lower() == "md":
        metrics = "GFLOPs"
    elif benchmark.lower() == "correlator":
        metrics = "GFLOPs"
    db_queue.execute("SELECT COUNT(id) FROM " + table + " WHERE " + scenario)
    items = db_queue.fetchall()
    nr_items = items[0][0]
    if int(nr_items) == 0:
        return [-1]
    db_queue.execute("SELECT MIN(" + metrics + ") FROM " + table + " WHERE " + scenario)
    items = db_queue.fetchall()
    results.append(int(items[0][0]))
    for quartile in range(1, 4):
        db_queue.execute("SELECT " + metrics + " FROM " + table + " WHERE (" + scenario + ") ORDER BY " + metrics + " LIMIT " + str(int(nr_items / 4) * quartile) + ",1")
        items = db_queue.fetchall()
        results.append(int(items[0][0]))
    db_queue.execute("SELECT MAX(" + metrics + ") FROM " + table + " WHERE " + scenario)
    items = db_queue.fetchall()
    results.append(int(items[0][0]))
    return results

def get_histogram(db_queue, table, benchmark, scenario):
    """Returns the performance histogram of the configurations."""
    results = dict()
    metrics = ""
    if benchmark.lower() == "triad":
        metrics = "GBs"
    elif benchmark.lower() == "reduction":
        metrics = "GBs"
    elif benchmark.lower() == "stencil":
        metrics = "GFLOPs"
    elif benchmark.lower() == "md":
        metrics = "GFLOPs"
    elif benchmark.lower() == "correlator":
        metrics = "GFLOPs"
    db_queue.execute("SELECT " + metrics + " FROM " + table + " WHERE (" + scenario + ") ORDER BY " + metrics)
    items = db_queue.fetchall()
    for item in items:
        value = int(item[0])
        if value not in results:
            results[value] = 1
        else:
            results[value] = results[value] + 1
    return results

