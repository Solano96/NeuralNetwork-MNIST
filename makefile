SRC = src
INC = include
OBJ = obj

CXX = g++
CXXFLAGS =  -Wall -Wextra -g  -I$(INC)

main: $(OBJ)/main.o $(OBJ)/readMNIST.o $(OBJ)/perceptronSimple.o
	g++ -fopenmp -o main -O2 $(OBJ)/main.o $(OBJ)/readMNIST.o $(OBJ)/perceptronSimple.o `pkg-config --cflags --libs opencv`

$(OBJ)/main.o: $(SRC)/main.cpp $(INC)/readMNIST.h $(INC)/perceptronSimple.h
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/main.o $(SRC)/main.cpp

$(OBJ)/readMNIST.o: $(INC)/readMNIST.h $(SRC)/readMNIST.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/readMNIST.o $(SRC)/readMNIST.cpp

$(OBJ)/perceptronSimple.o: $(INC)/perceptronSimple.h $(SRC)/perceptronSimple.cpp
	$(CXX) -c $(CXXFLAGS) -o $(OBJ)/perceptronSimple.o $(SRC)/perceptronSimple.cpp

.PHONY: clean
clean:
	rm -rf $(OBJ)/*.o
