# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra
EXEC = test3

$(EXEC): test3.o Layer_version3.o
	$(CXX) $(CXXFLAGS) -o $(EXEC) test3.o Layer_version3.o


Layer_version3.o: Layer_version3.cpp Layer_version3.hpp
	$(CXX) $(CXXFLAGS) -c Layer_version3.cpp -o Layer_version3.o

test3.o: test3.cpp
	$(CXX) $(CXXFLAGS) -c test3.cpp -o test3.o

clean:
	rm -f Layer_version3.o test3.o $(EXEC)