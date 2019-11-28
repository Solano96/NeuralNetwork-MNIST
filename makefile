SRC = src
INC = include
OBJ = obj

CXX = g++ -std=c++11
CXXFLAGS =  -Wall -Wextra -g -I$(SRC) -I$(INC)

main: $(OBJ)/main.o $(OBJ)/activations.o $(OBJ)/utils.o $(OBJ)/readMNIST.o $(OBJ)/perceptronSimple.o $(OBJ)/perceptronMultiLayer.o
	g++ -fopenmp -o main -O2 $(OBJ)/main.o $(OBJ)/readMNIST.o $(OBJ)/perceptronSimple.o $(OBJ)/perceptronMultiLayer.o $(OBJ)/activations.o $(OBJ)/utils.o

$(OBJ)/main.o: $(SRC)/main.cpp $(INC)/readMNIST.h $(INC)/perceptronSimple.h
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/main.o -O2 $(SRC)/main.cpp

$(OBJ)/readMNIST.o: $(INC)/readMNIST.h $(SRC)/readMNIST.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/readMNIST.o -O2 $(SRC)/readMNIST.cpp

$(OBJ)/perceptronSimple.o: $(INC)/perceptronSimple.h $(SRC)/perceptronSimple.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/perceptronSimple.o -O2 $(SRC)/perceptronSimple.cpp

$(OBJ)/perceptronMultiLayer.o: $(INC)/perceptronMultiLayer.h $(SRC)/perceptronMultiLayer.cpp $(INC)/activations.h $(INC)/utils.h
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/perceptronMultiLayer.o -O2 $(SRC)/perceptronMultiLayer.cpp

$(OBJ)/activations.o: $(INC)/activations.h $(SRC)/activations.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/activations.o -O2 $(SRC)/activations.cpp

$(OBJ)/utils.o: $(INC)/utils.h $(SRC)/utils.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/utils.o -O2 $(SRC)/utils.cpp

.PHONY: clean
clean:
	rm -rf $(OBJ)/*.o
