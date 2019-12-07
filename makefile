SRC = src
INC = include
OBJ = obj

CXX = g++ -std=c++17
CXXFLAGS = -O2 -fopenmp -Wall -Wextra -g -I$(SRC) -I$(INC) -march=native

main: $(OBJ)/main.o $(OBJ)/readMNIST.o $(OBJ)/perceptronSimple.o $(OBJ)/perceptronMultiLayer.o $(OBJ)/convolutionalNeuralNetwork.o $(OBJ)/convolutionFunctions.o
	g++ -o main $(CXXFLAGS) $(OBJ)/main.o $(OBJ)/readMNIST.o $(OBJ)/perceptronSimple.o $(OBJ)/perceptronMultiLayer.o $(OBJ)/convolutionalNeuralNetwork.o $(OBJ)/convolutionFunctions.o

$(OBJ)/main.o: $(SRC)/main.cpp $(INC)/readMNIST.h
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/main.o $(SRC)/main.cpp

$(OBJ)/readMNIST.o: $(INC)/readMNIST.h $(SRC)/readMNIST.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/readMNIST.o $(SRC)/readMNIST.cpp

$(OBJ)/perceptronSimple.o: $(INC)/perceptronSimple.h $(SRC)/perceptronSimple.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/perceptronSimple.o $(SRC)/perceptronSimple.cpp

$(OBJ)/perceptronMultiLayer.o: $(INC)/perceptronMultiLayer.h $(SRC)/perceptronMultiLayer.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/perceptronMultiLayer.o $(SRC)/perceptronMultiLayer.cpp

$(OBJ)/convolutionalNeuralNetwork.o: $(INC)/convolutionalNeuralNetwork.h $(SRC)/convolutionalNeuralNetwork.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/convolutionalNeuralNetwork.o $(SRC)/convolutionalNeuralNetwork.cpp

$(OBJ)/convolutionFunctions.o: $(INC)/convolutionFunctions.h $(SRC)/convolutionFunctions.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/convolutionFunctions.o $(SRC)/convolutionFunctions.cpp

.PHONY: clean
clean:
	rm -rf $(OBJ)/*.o
