CXX = g++
OBJS = main.o Layer.o Matrix.o ActivationFcts.o LossFcts.o Optimizers.o 
EXEC = nnproj

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $(EXEC)


clean: 
	rm $(OBJS) $(EXEC)