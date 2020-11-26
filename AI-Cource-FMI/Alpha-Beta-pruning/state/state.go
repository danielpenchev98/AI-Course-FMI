package state

import (
    "strings"
    "fmt"
  	"math/rand"
	"time"
	"errors"
)

const maxDepth = 9

type Histogram struct {
    rows      [][]int
    cols      [][]int
    diags     [][]int //every diagonal has 2 values -> first one is number of tiles placed by first player on the diagonal, the second one is for the second player
}

type State struct {
    freeTiles map[int]struct{}
	board     [][]int
	winner int //-1 noone yet or draw, 0 second player, 1 first player
    Histogram
}

var emptyStruct struct{}

func CreateState() *State{
    state := State{
        freeTiles: make(map[int]struct{}),
        board: [][]int{
            {-1, -1, -1},
            {-1, -1, -1},
            {-1, -1, -1},
		},
		winner: -1,
        Histogram: Histogram{
            rows:  [][]int{
                {0, 0},
                {0, 0},
                {0, 0},
            },
            cols:  [][]int{
                {0, 0},
                {0, 0},
                {0, 0},
            },
            diags: [][]int{
                {0, 0},
                {0, 0},
                {0, 0},
            },
        },
    }

    for i := 0; i < 9; i++ {
        state.freeTiles[i] = emptyStruct
    }

    return &state
}

func (s *State) PlayMove(playerNumber int, y int, x int) {
    s.board[y][x] = playerNumber
    delete(s.freeTiles, y*3+x)
    
    s.rows[y][playerNumber]++
    s.cols[x][playerNumber]++
    if x == y {
        s.diags[0][playerNumber]++
    }
    if x+y == 2 {
        s.diags[1][playerNumber]++
	}
	
	if s.rows[y][playerNumber] == 3 || s.cols[x][playerNumber] == 3 || s.diags[0][playerNumber]==3 || s.diags[1][playerNumber]==3{
		s.winner = playerNumber
	}
}

func (s *State) UndoMove(playerNumber int, y int, x int) {
    s.board[y][x] = -1
    s.freeTiles[y*3+x] = emptyStruct
	
	if s.rows[y][playerNumber] == 3 || s.cols[x][playerNumber] == 3 || s.diags[0][playerNumber]==3 || s.diags[1][playerNumber]==3{
		s.winner = -1
	}

    s.rows[y][playerNumber]--
    s.cols[x][playerNumber]--
    if x == y {
        s.diags[0][playerNumber]--
    }
    if x+y == 2 {
        s.diags[1][playerNumber]--
    }
}

func (s *State) Utility() int{
    return s.getColsScore() + s.getRowsScore() + s.getDiagsScore()
}

func (s *State) getRowsScore() int {
    var score int
    for i:=0;i<len(s.board);i++{
        multiplier := -1
        for j:=0;j<2;j++{
            if s.rows[i][j] == 3 {
                score += 100 * multiplier
            } else if s.rows[i][j]==2 && s.rows[i][(j+1)%2]==0{
                score +=10 * multiplier
            }
            multiplier+=2
        }
        
        if s.rows[i][1] == 1 && s.rows[i][0] == 0{
            score += 1
        } else if s.rows[i][1] == 0 && s.rows[i][0] == 1 {
            score -=1
        }
    }
    return score
}

func (s *State) getColsScore() int {
    var score int
    for i:=0;i<len(s.board);i++{
        multiplier := -1
        for j:=0;j<2;j++{
            if s.cols[i][j] == 3 {
                score += 100 * multiplier
            } else if s.cols[i][j]==2 && s.cols[i][(j+1)%2]==0 {
                score +=10 * multiplier
            }
            multiplier+=2
        }
        
        if s.cols[i][1] == 1 && s.cols[i][0] == 0{
            score += 1
        } else if s.cols[i][1] == 0 && s.cols[i][0] == 1 {
            score -=1
        }
    }
    return score
}

func (s *State) getDiagsScore() int {
    var score int
    for i:=0 ; i< 2;i++{
        multiplier := -1
        for j:=0;j<2;j++{
            if s.diags[i][j] == 3 {
                score += 100 * multiplier
            } else if s.diags[i][j]==2 && s.diags[i][(j+1)%2]==0 {
                score +=10 * multiplier
            }
            multiplier+=2
        }

        if  s.diags[i][1]==1 && s.diags[i][0]==0{
            score +=1
        } else if s.diags[i][1]==0 && s.diags[i][0]==1 {
            score -=1
        }
    }
    return score
}

func (s *State) String() string {
    var sb strings.Builder
    for i := 0; i < len(s.board); i++ {
        for j := 0; j < len(s.board); j++ {
            sb.WriteString(fmt.Sprintf("%d ", s.board[i][j]))
        }
        sb.WriteString("\n")
    }
    return sb.String()
}

func abs(a int) int{
    if a>=0{
        return a
    }
    return -a
}

func (s *State) GetWinner() (int,error){
	if !s.HasWinner(){
		return -1, errors.New("The game doesnt have a winner")
	}
	return s.winner,nil
}

func (s *State) HasWinner() (bool) {
    return s.winner != -1
}

func (s *State) IsTerminal() bool {
    return len(s.freeTiles)==0 || s.HasWinner()
}

func (s *State) GetNeightbours() []int{
    neightbours := make([]int,len(s.freeTiles))
    iter :=0
    for key,_ := range s.freeTiles {
        neightbours[iter]=key
        iter++
    }
    rand.Seed(time.Now().UnixNano())
    rand.Shuffle(len(neightbours), func(i, j int) { neightbours[i], neightbours[j] = neightbours[j], neightbours[i] })
    return neightbours
}