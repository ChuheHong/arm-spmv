CURRENT_PATH = $(shell pwd)
CXX=g++
CXXFLAGS+=-O2 -lpthread -fopenmp
CXXFLAGS+=-DUSE_OPENMP -I$(CURRENT_PATH)/../include
OBJ_PATH =  $(CURRENT_PATH)/../obj
LIBS=-I$(CURRENT_PATH)../include
SRC := $(wildcard *.cpp)
OBJ := $(SRC:.cpp=.o)

RANLIB = /usr/bin/ranlib

all: TH_sparse.a

TH_sparse.a : $(OBJ)
	ar rc $@ $(addprefix $(OBJ_PATH)/, $^)
	$(RANLIB) $@
	mv TH_sparse.a $(CURRENT_PATH)/../bin

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $(OBJ_PATH)/$@

clean:
	rm -f $(OBJ_PATH)/*.o
	
.PHONY: clean
