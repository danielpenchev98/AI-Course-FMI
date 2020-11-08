#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>  
#include <random>
#include <chrono>
#include <list>
//#include <unordered_map>

int sizeOfBoardd;
int* queens;
int* rows;
int* diagonal1;
int* diagonal2;
int* container;

const int k = 1000;
std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

bool isInConflict(int q1, int q2) {
	return queens[q1] == queens[q2] ||
		q1 - queens[q1] == q2 - queens[q2] ||
		q1 + queens[q1] == q2 + queens[q2];
}

inline int getDiag1Index(const int x, const int y) {
	return x - y + sizeOfBoardd - 1;
}

inline int getDiag2Index(const int x, const int y) {
	return x + y;
}

int getConflicts(int x, int y) {
	return diagonal1[getDiag1Index(x, y)] +
		diagonal2[getDiag2Index(x, y)] +
		rows[y];
}

void initializeBoard() {

	for (int i = 0; i < 2 * sizeOfBoardd - 1; i++) {
		diagonal1[i] = 0;
		diagonal2[i] = 0;
	}

	for (int i = 0; i < sizeOfBoardd; i++) {
		rows[i] = 0;
	}

	queens[0] = rng() % sizeOfBoardd;
	rows[queens[0]]++;
	diagonal1[getDiag1Index(0,queens[0])]++;
	diagonal2[getDiag2Index(0,queens[0])]++;

	//int* conflicts = new int[sizeOfBoardd];

	for (int i = 1; i < sizeOfBoardd; i++) {
		int frontCollisions = INT32_MAX;
		int bestCount = 0;
		for (int k = 0; k < sizeOfBoardd; k++) {
			int collisions = getConflicts(i, k);

			//std::cout << "Queen [" << i << "] on row [" << k << "] will have conflicts [" << collisions << "]\n";

			if (bestCount == 0 || frontCollisions == collisions) {
				container[bestCount] = k;
				frontCollisions = collisions;
				bestCount++;
			}
			else if (frontCollisions > collisions) {
				frontCollisions = collisions;
				container[0] = k;
				bestCount = 1;
			}
		}

		//pick random from best possible scenarios
		int winner = rng() % bestCount;
		//std::cout << "Winner is [" << conflicts[winner] << "] from the other with the same result [" << bestCount << "]\n";

		queens[i] = container[winner];//conflicts[winner];


		//std::cout << "Should update diagonal1[" << getDiag1Index(i, queens[i]) << " and diagonal2 ["<<getDiag2Index(i, queens[i]) << "]\n";
		//Update conflict arrays
		diagonal1[getDiag1Index(i, queens[i])]++;
		diagonal2[getDiag2Index(i, queens[i])]++;
		rows[queens[i]]++;

		bestCount = 0;
	}

	//delete[] conflicts;
}


std::pair<int,int> getMaxConflictQueen() {
	//int* maxConflictQueens = new int[sizeOfBoardd];
	int frontConflicts = 0;
	int bestCount = 0;
	for (int i = 0; i < sizeOfBoardd; i++) {
		int conflicts = getConflicts(i,queens[i]) - 3;

		if (bestCount == 0 || frontConflicts == conflicts) {
			container[bestCount] = i;
			frontConflicts = conflicts;
			bestCount++;
		}
		else if (frontConflicts < conflicts) {
			frontConflicts = conflicts;
			container[0] = i;
			bestCount = 1;
		}
	}

	int winner = rng() % bestCount;

	//int toReturn = maxConflictQueens[winner];
	//delete[] maxConflictQueens;
	
	return std::pair<int,int>(container[winner],frontConflicts);
}

int getLeastConflictRow(const int queen) {
	int oldRow = queens[queen];
	int bestCount = 0;
	int frontCollisions = INT32_MAX;
	//int* positions = new int[sizeOfBoardd];
	for (int i = 0; i < sizeOfBoardd; i++) {
		if (oldRow == i) {
			continue;
		}
		
		int collisions = getConflicts(queen,i);

		if (bestCount == 0 || frontCollisions == collisions) {
			container[bestCount]=i;
			frontCollisions = collisions;
			bestCount++;
		}
		else if (frontCollisions > collisions) {
			frontCollisions = collisions;
			container[0] = i;
			bestCount = 1;
		}
	}

	int winner = rng() % bestCount;
	
	//int toReturn = positions[winner];

	//delete[] positions;
	return container[winner];
}

void updateConflictStatistics(const int queen, const int newRow) {
	rows[queens[queen]]--;
	diagonal1[getDiag1Index(queen, queens[queen])]--;
	diagonal2[getDiag2Index(queen, queens[queen])]--;

	queens[queen] = newRow;

	rows[newRow]++;
	diagonal1[getDiag1Index(queen, newRow)]++;
	diagonal2[getDiag2Index(queen, newRow)]++;
}

void minimumConflict() {
	int maxConflictQueen = -1;
	for (int i = 0; i <= k * sizeOfBoardd; i++) {
		const std::pair<int,int>& maxConflictQueen = getMaxConflictQueen();
		if (maxConflictQueen.second == 0) {
			std::cout << "Final state" << std::endl;
			for (int i = 0; i < sizeOfBoardd; i++) {
				std::cout << queens[i] << std::endl;
			}
			return;
		}
		int leastConflictRow = getLeastConflictRow(maxConflictQueen.first);

		updateConflictStatistics(maxConflictQueen.first, leastConflictRow);
	}

	std::cout << "restart" << std::endl;
	minimumConflict();
}

int main() {
	std::cin >> sizeOfBoardd;

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	queens = new int[sizeOfBoardd];
	rows = new int[sizeOfBoardd];
	container = new int[sizeOfBoardd];
	diagonal1 = new int[2*sizeOfBoardd-1];
	diagonal2 = new int[2*sizeOfBoardd-1];
	
	initializeBoard();

	//for (int i = 0; i < sizeOfBoardd; i++) {
	//	std::cout << queens[i] << " " << std::endl;
	//}

	/*std::pair<int,int> result = getMaxConflictQueen();
	for (int i = 0; i < sizeOfBoardd; i++) {
		std::cout << "Queen [" << i <<"] has conflicts" << getConflicts(i, queens[i]) << std::endl;
		std::cout << "Diagonal1 [" << getDiag1Index(i, queens[i]) << "] has conflicts [" << diagonal1[getDiag1Index(i, queens[i])] << std::endl;
		std::cout << "Diagonal2 [" << getDiag2Index(i, queens[i]) << "] has conflicts [" << diagonal2[getDiag1Index(i, queens[i])] << std::endl;
	}*/

	//std::cout << "Best queen [" << result.first << "] with conflicts " << result.second << std::endl;

	minimumConflict();
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << "Time consumed :" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

	return 0;
}