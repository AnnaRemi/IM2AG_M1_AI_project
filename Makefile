# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -O2 -std=c++17
OBJS = main.o Layer.o Matrix.o ActivationFcts.o LossFcts.o
EXEC = nnproj

# Default target
all: $(EXEC)

# Link the executable
$(EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $(EXEC)

# Compile object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean target
clean: 
	rm -f $(OBJS) $(EXEC)

