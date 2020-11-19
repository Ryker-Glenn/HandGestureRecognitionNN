#include <iostream>
#include <fstream>
#include <sstream>
#include "NN.h"
#include "Gesture.h"

using namespace std;

typedef vector<RowVector*> dat;
int main() {
	Gesture g;
	g.capture();
	return 0;
}