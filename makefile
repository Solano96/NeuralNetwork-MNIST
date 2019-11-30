SRC = src
INC = include
OBJ = obj

CXX = g++ -std=c++11
CXXFLAGS =  -Wall -Wextra -g -I$(SRC) -I$(INC) -march=native

main: $(OBJ)/main.o $(OBJ)/readMNIST.o $(OBJ)/perceptronSimple.o $(OBJ)/perceptronMultiLayer.o
	g++ -fopenmp -o main -O2 $(OBJ)/main.o $(OBJ)/readMNIST.o $(OBJ)/perceptronSimple.o $(OBJ)/perceptronMultiLayer.o

$(OBJ)/main.o: $(SRC)/main.cpp $(INC)/readMNIST.h $(INC)/perceptronSimple.h
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/main.o -O2 $(SRC)/main.cpp

$(OBJ)/readMNIST.o: $(INC)/readMNIST.h $(SRC)/readMNIST.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/readMNIST.o -O2 $(SRC)/readMNIST.cpp

$(OBJ)/perceptronSimple.o: $(INC)/perceptronSimple.h $(SRC)/perceptronSimple.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/perceptronSimple.o -O2 $(SRC)/perceptronSimple.cpp

$(OBJ)/perceptronMultiLayer.o: $(INC)/perceptronMultiLayer.h $(SRC)/perceptronMultiLayer.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/perceptronMultiLayer.o -O2 $(SRC)/perceptronMultiLayer.cpp


.PHONY: clean
clean:
	rm -rf $(OBJ)/*.o
