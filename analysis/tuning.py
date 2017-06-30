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
"""Functions to tune the data stored in the database."""

def tune(db_queue, table, benchmark, scenario):
    """Tune a selection of data from a table."""
    extra = ""
    metrics = ""
    if benchmark.lower() == "triad":
        extra = "vector,"
        metrics = "GBs,"
    elif benchmark.lower() == "reduction":
        extra = "nrItemsPerBlock,vector,"
        metrics = "GBs,"
    elif benchmark.lower() == "stencil":
        extra = "localMemory,"
        metrics = "GFLOPs,"
    elif benchmark.lower() == "md":
        metrics = "GFLOPs,"
    elif benchmark.lower() == "correlator":
        extra = "sequentialTime,parallelTime,constantMemory,width,height,"
        metrics = "GFLOPs,"
    elif benchmark.lower() == "blackscholes":
        extra = "vector,"
        metrics = "GFLOPs,"
    print("SELECT " + extra + "nrThreadsD0,nrThreadsD1,nrThreadsD2,nrItemsD0,nrItemsD1,nrItemsD2," + metrics + "time,time_err,variation FROM " + table + " WHERE (" + metrics.rstrip(",") + " = (SELECT MAX(" + metrics.rstrip(",") + ") FROM " + table + " WHERE (" + scenario + "))) AND (" + scenario + ")")
    db_queue.execute("SELECT " + extra + "nrThreadsD0,nrThreadsD1,nrThreadsD2,nrItemsD0,nrItemsD1,nrItemsD2," + metrics + "time,time_err,variation FROM " + table + " WHERE (" + metrics.rstrip(",") + " = (SELECT MAX(" + metrics.rstrip(",") + ") FROM " + table + " WHERE (" + scenario + "))) AND (" + scenario + ")")
    best = db_queue.fetchall()
    if len(best) > 0:
        return best[0]
    else:
        return [-1]
