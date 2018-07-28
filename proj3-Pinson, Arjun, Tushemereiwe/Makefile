NVCC        := nvcc -g
CC          := g++

EXECUTABLE  := cdpQuadtree
CUFILES     := $(wildcard *.cu)
HEADERS     := $(wildcard *.h)
# COBJ        := $(patsubst %.cpp,%.o,$(wildcard *.cpp))
MAINSRC     := cdpQuadtree.cu

$(EXECUTABLE): $(HEADERS) $(CUFILES) $(COBJ)
	$(NVCC) $(FLAGS) $(INCLUDES) -o $@ $(MAINSRC)

clean:
	rm -f $(EXECUTABLE) *.o
