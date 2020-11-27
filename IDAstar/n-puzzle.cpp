#include <iostream>
#include <tuple>
#include <cmath>
#include <stack>
#include <chrono>
#include <list>

enum Move { up, down, right, left, none };
int sizeOfBoard;
std::pair<int, int> emptyTileTargetPos;
const char numberOfNeightbours = 4;
std::pair<int, int>* targetPos;
std::stack<Move> history;
long long emptyTileNumber;

inline Move getOppositeMove(const Move move) {
	switch (move) {
	case up: return down;
	case down: return up;
	case right: return left;
	case left: return right;
	default: return none;
	}
}

const std::pair<int, int> getTargetPos(const int tileIndex) {
	if (tileIndex == -2) {
		return std::pair<int, int>(sizeOfBoard - 1, sizeOfBoard - 1);
	}

	const int row = std::ceil(tileIndex / sizeOfBoard);
	const int col = tileIndex % sizeOfBoard;
	return std::pair<int, int>(row, col);
}


void setUpTargetPosTable(const std::pair<int, int>& emptyTilePos) {
	targetPos = new std::pair<int, int>[sizeOfBoard * sizeOfBoard];
	int emptyTileNumber = emptyTileTargetPos.first * sizeOfBoard + emptyTileTargetPos.second + 1;
	for (int i = 1; i < sizeOfBoard * sizeOfBoard; i++) {
		int offset = i < emptyTileNumber ? -1 : 0;
		targetPos[i] = getTargetPos(i + offset);
	}
}

inline const std::pair<const int, const int> nextTile(const std::pair<const int, const int>& emptyTile, const Move move) {
	std::pair<int, int> next(emptyTile);
	switch (move) {
	case up: next.first += 1; break;
	case down: next.first -= 1; break;
	case right: next.second -= 1; break;
	case left: next.second += 1; break;
	}
	return next;
}

const std::pair<int, int> getEmptyTilePos(int** board) {
	for (int i = 0; i < sizeOfBoard; i++) {
		for (int j = 0; j < sizeOfBoard; j++) {
			if (board[i][j] == 0) {
				return std::pair<int, int>(i, j);
			}
		}
	}
	return std::pair<int, int>(-1, -1);
}

inline int updateManhattan(int** board, const std::pair<int, int>& emptyTileCurrentPos, const std::pair<int, int>& emptyTilePreviousPos, const int parentManhattan) {
	if (emptyTileCurrentPos == emptyTilePreviousPos) {
		return parentManhattan;
	}
	const int number = board[emptyTilePreviousPos.first][emptyTilePreviousPos.second];
	const std::pair<int, int>& target = targetPos[number];
	return parentManhattan - (abs(emptyTileCurrentPos.first - target.first) + abs(emptyTileCurrentPos.second - target.second)) +
		abs(emptyTilePreviousPos.first - target.first) + abs(emptyTilePreviousPos.second - target.second);
}

int calculateManhattan(int** board) {
	int distance = 0;
	for (int i = 0; i < sizeOfBoard; i++) {
		for (int j = 0; j < sizeOfBoard; j++) {
			if (board[i][j] == 0) continue;
			const std::pair<int, int>& target = targetPos[board[i][j]];
			distance += abs(i - target.first) + abs(j - target.second);
		}
	}
	return distance;
}

long long countInversions(int** board) {
	long long inversions = 0;
	int temp = 0;
	int* brd = new int[sizeOfBoard * sizeOfBoard];
	for (int i = 0; i < sizeOfBoard; i++) {
		for (int j = 0; j < sizeOfBoard; j++) {
			brd[temp] = board[i][j];
			temp++;
		}
	}

	for (int i = 0; i < sizeOfBoard * sizeOfBoard; i++) {
		for (int j = i + 1; j < sizeOfBoard * sizeOfBoard; j++) {
			if (brd[j] < brd[i] && brd[i] != 0 && brd[j] != 0) {
				inversions++;
			}
		}
	}

	delete[] brd;
	return inversions;
}

bool isSolvable(int** start) {
	int emptyTileRowStart = -1;
	for (int i = 0; i < sizeOfBoard; i++) {
		for (int j = 0; j < sizeOfBoard; j++) {
			if (start[i][j] == 0) {
				emptyTileRowStart = i;
			}
		}
	}

	long long parityStart = countInversions(start) + (sizeOfBoard % 2 == 0 ? emptyTileRowStart : 0);
	long long parityEnd = sizeOfBoard % 2 == 0 ? emptyTileTargetPos.first : 0;

	return parityStart % 2 == parityEnd % 2;
}


inline bool isTileValid(const std::pair<int, int>& tilePos) {
	return (tilePos.first >= 0 && tilePos.first < sizeOfBoard) &&
		(tilePos.second >= 0 && tilePos.second < sizeOfBoard);
}

inline void swapTile(int** board, const std::pair<int, int>& a, const std::pair<int, int>& b) {
	const int temp = board[a.first][a.second];
	board[a.first][a.second] = board[b.first][b.second];
	board[b.first][b.second] = temp;
}

inline int search(int** board, const std::pair<int, int>& emptyTilePos, const int path, const int threshold, const Move prevMove, const int parentManhattan) {
	const Move oppositePrevMove = getOppositeMove(prevMove);
	const std::pair<int, int>& prevEmptyTilePos = nextTile(emptyTilePos, oppositePrevMove);
	const int heuristic = updateManhattan(board, emptyTilePos, prevEmptyTilePos, parentManhattan);
	const int cost = heuristic + path;

	if (cost > threshold) {
		return cost;
	}
	else if (heuristic == 0) {
		history.push(prevMove);
		std::cout << path << std::endl;
		return 0;
	}

	int min = INT32_MAX;
	for (int i = 0; i < numberOfNeightbours; i++) {
		if (oppositePrevMove == i) {
			continue;
		}

		const std::pair<int, int>& neightbour = nextTile(emptyTilePos, static_cast<Move>(i));
		if (!isTileValid(neightbour)) {
			continue;
		}

		swapTile(board, neightbour, emptyTilePos);
		const int temp = search(board, neightbour, path + 1, threshold, static_cast<Move>(i), heuristic);
		swapTile(board, neightbour, emptyTilePos);

		if (temp == 0) {
			history.push(prevMove);
			return 0;
		}
		else if (temp < min) {
			min = temp;
		}
	}
	return min;
}

void IDA(int** startBoard) {
	const std::pair<const int, const int>& emptyTilePos = getEmptyTilePos(startBoard);
	const int startHeuristic = calculateManhattan(startBoard);
	int threshold = startHeuristic;
	while (true) {
		//std::cout << "threshold :" << threshold << std::endl;
		int answer = search(startBoard, emptyTilePos, 0, threshold, none, startHeuristic);
		if (answer == 0) {
			break;
		}
		threshold = answer;
	}
}

int main() {
	int numberOfTiles = -1;
	std::cin >> numberOfTiles;
	sizeOfBoard = sqrt(numberOfTiles + 1);

	int emptyTileNumber = -1;
	std::cin >> emptyTileNumber;
	emptyTileTargetPos = getTargetPos(emptyTileNumber - 1);

	int** board = new int* [sizeOfBoard];
	for (int i = 0; i < sizeOfBoard; i++) {
		board[i] = new int[sizeOfBoard];
		for (int j = 0; j < sizeOfBoard; j++) {
			std::cin >> board[i][j];
		}
	}

	if (!isSolvable(board)) {
		std::cout << "The puzzle is unsolvable." << std::endl;
		return 0;
	}
	setUpTargetPosTable(emptyTileTargetPos);

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	IDA(board);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << "Time consumed :" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

	while (!history.empty()) {
		switch (history.top()) {
		case up: std::cout << "up"; break;
		case down: std::cout << "down"; break;
		case right: std::cout << "right"; break;
		case left: std::cout << "left"; break;
		}
		history.pop();
		if (!history.empty()) {
			std::cout << std::endl;
		}

	}

	for (int i = 0; i < sizeOfBoard; i++) {
		delete[] board[i];
	}
	delete[] board;
	return 0;
}
