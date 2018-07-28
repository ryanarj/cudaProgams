/*

	Project 3
	Summer 2018
	Brian Pinson
	Karshan Arjun
	Mark Tushemereiwe 


*/

Program is designed to print a histogram computed by the CPU and the same histogram computed by CUDA traversing the nodes of a Quadtree.

compile: "make"
run: "./cdpQuadtree num_of_points size_of_nodes num_of_levels bucket_size block_size"

examples:
./cdpQuadtree 10000 100 10
./cdpQuadtree 20000 200 10 1000
./cdpQuadtree 30000 300 10 500 64

Make sure the size of nodes and levels are large enough to fit all the points in the quadtree because the CPU histogram is created from the random number generator not the tree.
