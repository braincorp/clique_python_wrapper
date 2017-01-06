/*
Notice that cliquer is under GPL. See LICENCE in src/cliquer

Wrapper code is based on Matlab c wrapper from here http://rehg.org/mht/
License:

Multiple Hypothesis Tracking Revisited
Copyright (c) 2015 Chanho Kim, Fuxin Li, Arridhana Ciptadi, James M. Rehg

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Numpy wrapper is based on:
Michael Droettboom wrapper, Licensed under the BSD license.

*/

#include <boost/python.hpp>
#include <numpy_boost_python.hpp>
#include <boost/format.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>
#include <cmath>
#include <limits>
#include <queue>

#include <math.h>
#include <stdio.h>

extern "C" {
#include "set.h"
#include "cliquer.h"
}


namespace py = boost::python;

void translate_exception(std::runtime_error const& e)
{
    // Use the Python 'C' API to set up an exception object
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

/* Output Arguments */
#define OUTPUT 	plhs[0]

#if !defined(MAX)
#define MAX(A, B) 	((A) > (B) ? (A) : (B))
#endif

#if !defined(MIN)
#define MIN(A, B) 	((A) < (B) ? (A) : (B))
#endif

graph_t* MatrixToGraph(const numpy_boost<uint8_t, 2> &adjacencyMatrix, const numpy_boost<int, 1> &weights)
{
    graph_t *ptrGraph = graph_new(weights.shape()[0]);

    for (int i = 0; i < adjacencyMatrix.shape()[0]; i++) {
      for (int j =i+1; j < adjacencyMatrix.shape()[1]; j++) {
        if (adjacencyMatrix[i][j] > 0) {
          GRAPH_ADD_EDGE(ptrGraph, i, j);
        }
      }
    }

    for (int i = 0; i < weights.shape()[0]; i++) {
      ptrGraph->weights[i] = weights[i];
    }

    /* Just to be cautios, ensure that we've produced a valid graph. */
    ASSERT(graph_test(ptrGraph, NULL));

    return ptrGraph;
}


set_t FindCliques(graph_t *ptrGraph, int iMinWeight, int iMaxWeight,
		int bOnlyMaximal, int iMaxNumCliques,
		int iCliqueListLength)
{
    int            iNumCliques;
    clique_options localopts;

    /* Set the clique_options.  These fields should all be null except
     * 'clique_list' and 'clique_list_length', which store the list of
     * cliques and the maximum length of the list of cliques, respectively. */
    localopts.time_function      = NULL;
    localopts.reorder_function   = NULL;
    localopts.reorder_map        = NULL;
    localopts.clique_list        = NULL;
    localopts.clique_list_length = 0;
    localopts.user_function      = NULL;
    localopts.user_data          = NULL;

    /* Find a single maximal clique in this graph of
     * minimum size 1 (argument 2) and no maximum size (argument 3). */

    set_t maxWeightedClique;
    maxWeightedClique = clique_find_single(ptrGraph, iMinWeight, iMaxWeight, bOnlyMaximal,
				  &localopts);

    return maxWeightedClique;
}

numpy_boost<uint8_t, 1>
find_max_weighted_clique(
  const numpy_boost<uint8_t,2> &adjacencyMatrix,
  const numpy_boost<int,1> & weightVector,
  int iMinWeight,
  int iMaxWeight,
  bool boolOnlyMaximal,
  int iMaxNumCliques
  ) {

    if (weightVector.shape()[0] != adjacencyMatrix.shape()[0]) {
      throw std::runtime_error("Non-consistent dimensions of weights and adjacency matrix");
    }

   if (adjacencyMatrix.shape()[0] != adjacencyMatrix.shape()[1]) {
      throw std::runtime_error("Adjacency matrix is not square");
    }

    /* Retrieve and store the size of the input adjacency matrix for size
     * checking and later use. */
    int iCols = weightVector.shape()[0];

    /* Declare variables to hold the input arguments (except the first).  Each
     * of these can be retrieved by indexing `prhs`. */
    iMinWeight     = MAX(0, iMinWeight);
    iMaxWeight     = MIN(iCols, iMaxWeight);
    int iOnlyMaximal   = boolOnlyMaximal ? TRUE: FALSE;

    /* Miscellaneous variable declarations. */
    int iNumCliquesReturned;

    /* Create a graph from the adjacency matrix `prhs[0]`. */
    graph_t  *ptrGraph = MatrixToGraph(adjacencyMatrix, weightVector);

    /* Find the cliques in the associated graph. */
    set_t arrCliqueList = FindCliques(
      ptrGraph, iMinWeight, iMaxWeight, iOnlyMaximal,
			iMaxNumCliques, iMaxNumCliques);

    /* We are done with the graph.  Free the memory used to store it. */
    graph_free(ptrGraph);

    /* Retrieve the number of cliques returned by the function, which is bounded
     * above by `iMaxNumCliques`. */
    iNumCliquesReturned = 1;

    /* Create the output matrix, which will have one row for each clique and
     * one column for each node of the graph. */
    numpy_boost<uint8_t, 1> output({iCols});
    /* Fill in the entries of this row by looping through the corresponding
     * clique to find the vertices contained in the clique. */
    for (int j = 0; j < iCols; j++)
    {
        if (SET_CONTAINS(arrCliqueList, j))
        {
          output[j] = 1;
        } else {
          output[j] = 0;
        }
    }

    /* Now that we've stored this clique as a row in a matrix, we can free
     * the memory used to store the clique. */
    set_free(arrCliqueList);

    return output;
}


BOOST_PYTHON_MODULE(_wrapper_max_weighted_clique_module)
{
    import_array();
    numpy_boost_python_register_type<uint8_t, 1>();
    numpy_boost_python_register_type<uint8_t, 2>();
    numpy_boost_python_register_type<int32_t, 1>();
    numpy_boost_python_register_type<int32_t, 2>();
    numpy_boost_python_register_type<double, 1>();
    numpy_boost_python_register_type<double, 2>();
    numpy_boost_python_register_type<int, 1>();
    numpy_boost_python_register_type<int, 2>();

    py::register_exception_translator<std::runtime_error>(&translate_exception);

    py::def("find_max_weighted_clique", find_max_weighted_clique);
}
