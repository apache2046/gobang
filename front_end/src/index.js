const { h, render, Component} = require("preact");
const {useState, useEffect} = require('preact/hooks');

const { Goban } = require("@sabaki/shudan");
const axios = require('axios').default

// const signMap = [
//   [0, 0, 0, -1, -1, -1, 1, 0, 1, 1, -1, -1, 0, -1, 0, -1, -1, 1, 0],
//   [0, 0, -1, 0, -1, 1, 1, 1, 0, 1, -1, 0, -1, -1, -1, -1, 1, 1, 0],
//   [0, 0, -1, -1, -1, 1, 1, 0, 0, 1, 1, -1, -1, 1, -1, 1, 0, 1, 0],
//   [0, 0, 0, 0, -1, -1, 1, 0, 1, -1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
//   [0, 0, 0, 0, -1, 0, -1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
//   [0, 0, -1, 0, 0, -1, -1, 1, 0, -1, -1, 1, -1, -1, 0, 1, 0, 0, 1],
//   [0, 0, 0, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1],
//   [0, 0, -1, 1, 1, 0, 1, -1, -1, 1, 0, 1, -1, 0, 1, -1, -1, -1, 1],
//   [0, 0, -1, -1, 1, 1, 1, 0, -1, 1, -1, -1, 0, -1, -1, 1, 1, 1, 1],
//   [0, 0, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1],
//   [-1, -1, -1, -1, 1, 1, 1, -1, 0, -1, 1, -1, -1, 0, -1, 1, 1, -1, 0],
//   [-1, 1, -1, 0, -1, -1, -1, -1, -1, -1, 1, -1, 0, -1, -1, 1, -1, 0, -1],
//   [1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 0, 1, -1, 0, -1, 1, -1, -1, 0],
//   [0, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1],
//   [0, 0, -1, 1, 0, 0, 1, 1, -1, -1, 0, 1, -1, 1, -1, 1, -1, 0, -1],
//   [0, 0, 1, 0, 1, 0, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 0],
//   [0, 0, 0, 0, 1, 1, 0, 1, -1, 0, -1, -1, 1, 1, 1, 1, -1, -1, -1],
//   [0, 0, 1, 1, -1, 1, 1, -1, 0, -1, -1, 1, 1, 1, 1, 0, 1, -1, 1],
//   [0, 0, 0, 1, -1, -1, -1, -1, -1, 0, -1, -1, 1, 1, 0, 1, 1, 1, 0],
// ];

class App extends Component {
  constructor(props) {
    super(props);

    let board_size = 11
    let signMap = new Array(board_size);
    for (var i = 0; i < board_size; i++) {
      signMap[i] = new Array(board_size);
      // for(var j = 0; j< board_size; j++)
      //   signMap[i][j] = 0;
    }
    // let signMap = [
    //   [0,0,0,0,0,0,0,0],
    //   [0,0,0,0,0,0,0,0],
    //   [0,0,0,0,0,0,0,0],
    //   [0,0,0,0,0,0,0,0],
    //   [0,0,0,0,0,0,0,0],
    //   [0,0,0,0,0,0,0,0],
    //   [0,0,0,0,0,0,0,0],
    //   [0,0,0,0,0,0,0,0],
    // ]
    this.state = {
      board: signMap,
      vertexSize: 36,//24,
      showCoordinates: true,
      alternateCoordinates: false,
      showCorner: false,
      showDimmedStones: false,
      fuzzyStonePlacement: false,
      animateStonePlacement: false,
      showPaintMap: false,
      showHeatMap: false,
      showMarkerMap: false,
      showGhostStones: false,
      showLines: false,
      showSelection: false,
      isBusy: true,
      init: false,
    };

    this.audio_pachi = []
    for (var i=0; i<4; i++){
      this.audio_pachi.push(new Audio(`/audio/${i}.mp3`))
    }

  }
  async playSound(){
    let index = Math.floor(Math.random() * this.audio_pachi.length)
    this.audio_pachi[index].play()
  }
  async initBoard() {
    const result = await axios({
      method: 'post',
      url: '/boardsize',
      data: {
        size: 11,
      }
    });
    const result2 = await axios({
      method: 'post',
      url: '/clearboard',
    });
    this.setState({'isBusy':false, init:true})
    console.log('end axio')
  };
  async play(actor, pos) {
    this.playSound()
    const result = await axios({
      method: 'post',
      url: '/play',
      data: {
        actor,
        pos,
      }
    });
    const rival_pos = await this.genmove(-1)
    let newboard = this.state.board
    console.log('rival_pos', rival_pos)
    newboard[rival_pos[1]][rival_pos[0]] = -1
    this.setState({'isBusy':false, board:newboard})
    this.playSound()
    console.log('end play')
  };
  async genmove(actor) {
    const result = await axios({
      method: 'post',
      url: '/genmove',
      data: {
        actor,
      }
    });
    this.setState({'isBusy':false})
    console.log('end genmove')
    return result.data.pos
  };

  render() {
    let {
      vertexSize,
      board,
      isBusy,
      showCoordinates,
      alternateCoordinates,
      showCorner,
      showDimmedStones,
      fuzzyStonePlacement,
      animateStonePlacement,
      showPaintMap,
      showHeatMap,
      showMarkerMap,
      showGhostStones,
      showLines,
      showSelection,
      init,
    } = this.state;

    console.log('in render')


    if(!init)
      this.initBoard();

    console.log('in render2')
    return h(
      "section",
      {
        style: {
          display: "grid",
          gridTemplateColumns: "15em auto",
          gridColumnGap: "1em",
        },
      },

      h(
        "div",
        {},
        h(Goban, {
          innerProps: {
            onContextMenu: (evt) => evt.preventDefault(),
          },

          vertexSize,
          animate: true,
          busy: isBusy,
          coordX: alternateCoordinates ? (i) => chineseCoord[i] : undefined,
          coordY: alternateCoordinates ? (i) => i + 1 : undefined,

          signMap: board,
          showCoordinates: true,
          // fuzzyStonePlacement,
          // animateStonePlacement,
          // paintMap: showPaintMap && paintMap,
          // heatMap: showHeatMap && heatMap,
          // markerMap: showMarkerMap && markerMap,
          // ghostStoneMap: showGhostStones && ghostStoneMap,

          onVertexMouseUp: (evt, [x, y]) => {
            let sign = evt.button === 0 ? 1 : -1;
            //   let newBoard = this.state.board.makeMove(sign, [x, y])

            //   this.setState({board: newBoard})
            board[y][x] = 1
            this.setState({board: board, isBusy:true})
            this.play(1, [x, y])
          },
        })
      )
    );
  }
}

render(h(App), document.getElementById("root"));
