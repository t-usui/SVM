CXX  = g++
OBJS  = main.o svc.o svm.o
CFLAGS  = -Wall -O3
DEBUG = -g
VPATH  = src

svm: $(OBJS)
	$(CXX) -o $@ $(OBJS) $(CFLAGS) $(DEBUG)

.cc.o:
	$(CXX) -c $< $(CFLAGS) $(DEBUG)

.PHONY: clean
clean:
	rm -f svm *.o
