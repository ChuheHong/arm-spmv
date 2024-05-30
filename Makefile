CXX=g++
CXXFLAGS+=-O2 -lpthread -fopenmp
CXXFLAGS+=-DUSE_OPENMP -I./include

LIBS=-I./include bin/TH_sparse.a
RANLIB = /usr/bin/ranlib

all: main 

main:
	$(CXX) $(CXXFLAGS) main.cpp -o bin/main $(LIBS)
	
%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $<

clean:
	rm -f *.o ./bin/main
	
.PHONY: clean
