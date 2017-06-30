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
"""Functions to manage the tables containing the data, and the data."""

def print_tuples(results):
    """Print the tuples returned by the database."""
    for item in results:
        print(item, end=" ")
    print()

def list_tables(db_queue):
    """Returns a list of the tables in the database."""
    db_queue.execute("SHOW TABLES")
    return db_queue.fetchall()

def list_values(db_queue, table, field, scenario):
    """Returns the list of unique values for a table's field."""
    if scenario != "":
        db_queue.execute("SELECT DISTINCT " + field + " FROM " + table + " WHERE (" + scenario + ") ORDER BY " + field)
    else:
        db_queue.execute("SELECT DISTINCT " + field + " FROM " + table + " ORDER BY " + field)
    return db_queue.fetchall()

def create_table(db_queue, table, scenario, extra, metrics):
    """Create a table to store the tuning data in the database."""
    db_queue.execute("CREATE TABLE " +  table + "(id INTEGER NOT NULL PRIMARY KEY AUTO_INCREMENT, " + scenario + " " + extra + " nrThreadsD0 INTEGER NOT NULL, nrThreadsD1 INTEGER NOT NULL, nrThreadsD2 INTEGER NOT NULL, nrItemsD0 INTEGER NOT NULL, nrItemsD1 INTEGER NOT NULL, nrItemsD2 INTEGER NOT NULL, " + metrics + " time FLOAT UNSIGNED NOT NULL, time_err FLOAT UNSIGNED NOT NULL, variation FLOAT UNSIGNED NOT NULL)")

def delete_table(db_queue, table):
    """Delete a table from the database."""
    db_queue.execute("DROP TABLE " + table)

def load_file(db_queue, table, input_file, benchmark):
    """Load a file containing auto-tuning values into a table."""
    for line in input_file:
        if str(line[0]).isalnum() and line[0] != "#":
            tokens = line.split(sep=" ")
            if benchmark.lower() == "triad":
                db_queue.execute("INSERT INTO " + table + " VALUES(NULL, " + tokens[0] + ", " + tokens[1] + ", " + tokens[2] + ", " + tokens[3] + ", " + tokens[4] + ", " + tokens[5]+ ", " + tokens[6] + ", " + tokens[7] + ", " + tokens[8] + ", " + tokens[9] + ", " + tokens[10] + ", " + tokens[11].rstrip("\n") + ")")
            elif benchmark.lower() == "reduction":
                db_queue.execute("INSERT INTO " + table + " VALUES(NULL, " + tokens[0] + ", " + tokens[1] + ", " + tokens[2] + ", " + tokens[3] + ", " + tokens[4] + ", " + tokens[5]+ ", " + tokens[6] + ", " + tokens[7] + ", " + tokens[8] + ", " + tokens[9] + ", " + tokens[10] + ", " + tokens[11] + ", " + tokens[12] + ", " + tokens[13].rstrip("\n") + ")")
            elif benchmark.lower() == "stencil":
                db_queue.execute("INSERT INTO " + table + " VALUES(NULL, " + tokens[0] + ", " + tokens[1] + ", " + tokens[2] + ", " + tokens[3] + ", " + tokens[4] + ", " + tokens[5]+ ", " + tokens[6] + ", " + tokens[7] + ", " + tokens[8] + ", " + tokens[9] + ", " + tokens[10] + ", " + tokens[11].rstrip("\n") + ")")
            elif benchmark.lower() == "md":
                db_queue.execute("INSERT INTO " + table + " VALUES(NULL, " + tokens[0] + ", " + tokens[1] + ", " + tokens[2] + ", " + tokens[3] + ", " + tokens[4] + ", " + tokens[5]+ ", " + tokens[6] + ", " + tokens[7] + ", " + tokens[8] + ", " + tokens[9] + ", " + tokens[10].rstrip("\n") + ")")
            elif benchmark.lower() == "correlator":
                db_queue.execute("INSERT INTO " + table + " VALUES(NULL, " + tokens[0] + ", " + tokens[1] + ", " + tokens[2] + ", " + tokens[3] + ", " + tokens[4] + ", " + tokens[5]+ ", " + tokens[6] + ", " + tokens[7] + ", " + tokens[8] + ", " + tokens[9] + ", " + tokens[10] + ", " + tokens[11] + ", " + tokens[12] + ", "  + tokens[13] + ", " + tokens[14] + ", " + tokens[15] + ", " + tokens[16] + ", " + tokens[17] + ", " + tokens[18] + ", " + tokens[19] + ", " + tokens[20].rstrip("\n") + ")")
            elif benchmark.lower() == "blackscholes":
                if tokens[0] == "inputSize" or tokens[0] == "OpenCL":
                    continue
                conf = tokens[2].split(sep=";")
                db_queue.execute("INSERT INTO " + table + " VALUES(NULL, " + tokens[0] + ", " + conf[0] + ", " + conf[1] + ", 0, 0, 0, 0, 0, " + tokens[9] + ", " + tokens[8] + ", " + tokens[10] + ", " + tokens[11] + ", " + tokens[12].rstrip("\n") + ")")
