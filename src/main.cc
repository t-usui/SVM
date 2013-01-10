#include "main.h"

int main(int argc, char **argv){
	classifier::SVC *svc = new classifier::SVC;

	svc->InputTrainingData();
	svc->InputTestData();

	svc->ClassifyTestData();

	delete svc;

	return 0;
}

