const { h, render, Component } = require("preact");
const { useState, useEffect } = require('preact/hooks');

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

    let board_size = 15

    let emptyBoard = new Array(board_size);
    for (let i = 0; i < board_size; i++) {
      emptyBoard[i] = new Array(board_size);
    }

    let emptyMarkerMap = new Array(board_size);
    for (let i = 0; i < board_size; i++) {
      emptyMarkerMap[i] = new Array(board_size);
      for (let j = 0; j < board_size; j++)
        emptyMarkerMap[i][j] = null
    }
    this.emptyMarkerMap = emptyMarkerMap
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
      board: emptyBoard,
      vertexSize: 36,//24,
      showCoordinates: true,
      alternateCoordinates: false,
      showCorner: false,
      showDimmedStones: false,
      fuzzyStonePlacement: false,
      animateStonePlacement: false,
      showPaintMap: false,
      showHeatMap: false,
      markerMap: emptyMarkerMap,
      showGhostStones: false,
      showLines: false,
      showSelection: false,
      isBusy: true,
      init: false,
    };

    this.audio_pachi = []
    for (let i = 0; i < 4; i++) {
      this.audio_pachi.push(new Audio(`/audio/${i}.mp3`))
    }

  }
  async playSound() {
    let index = Math.floor(Math.random() * this.audio_pachi.length)
    this.audio_pachi[index].play()
  }
  async initBoard() {
    const result = await axios({
      method: 'post',
      url: '/boardsize',
      data: {
        size: 15,
      }
    });
    const result2 = await axios({
      method: 'post',
      url: '/clearboard',
    });
    this.setState({ 'isBusy': false, init: true })
    console.log('end axio')
  };
  async placeStone(actor, pos) {
    let x = pos[0]
    let y = pos[1]

    this.playSound()
    let newBoard = this.state.board
    newBoard[y][x] = actor
    let newMarkerMap = structuredClone(this.emptyMarkerMap)
    newMarkerMap[y][x] = { type: 'point' }

    this.setState({ board: newBoard, markerMap: newMarkerMap })
  }
  async play(actor, pos) {
    const result = await axios({
      method: 'post',
      url: '/play',
      data: {
        actor,
        pos,
      }
    });
    console.log('end play', result)
    // let ret = {}
    // ret.actor = result.data[0]
    // ret.pos = result.data[1]
    // ret.v_actor = result.data[2]
    // ret.patterns = result.data[3]
    return result.data.data
  };
  async genmove(actor) {
    const result = await axios({
      method: 'post',
      url: '/genmove',
      data: {
        actor,
      }
    });
    // this.setState({ 'isBusy': false })
    console.log('end genmove')
    // let ret = {}
    // ret.actor = result.data[0]
    // ret.pos = result.data[1]
    // ret.v_actor = result.data[2]
    // ret.patterns = result.data[3]
    return result.data.data
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
      markerMap,
      showGhostStones,
      showLines,
      showSelection,
      init,
    } = this.state;

    console.log('in render')


    if (!init)
      this.initBoard();

    console.log('in render2')
    return h(
      "section",
      {
        style: {
          display: "grid",
          justifyContent: "center",
          // gridTemplateColumns: "15em auto",
          // gridColumnGap: "1em",
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
          markerMap,
          // ghostStoneMap: showGhostStones && ghostStoneMap,

          onVertexMouseUp: (evt, [x, y]) => {
            let sign = evt.button === 0 ? 1 : -1;
            //   let newBoard = this.state.board.makeMove(sign, [x, y])

            //   this.setState({board: newBoard})
            // board[y][x] = 1
            // this.setState({ board: board, isBusy: true })
            // this.play(1, [x, y])
            if (this.waiting)
              return
            this.waiting = true
            this.play(1, [x, y]).then(
              (data) => {
                let [actor, pos, v_actor, patterns ] = data
                this.placeStone(actor, pos)
                console.log(v_actor, patterns )
              }).then(() => {
                return this.genmove(-1).then(
                  (data) => {
                    let [ actor, pos, v_actor, patterns ] = data
                    this.placeStone(actor, pos)
                    console.log(v_actor, patterns )
                  }
                )
              }).then(() => { this.waiting = false })
          },
        })
      )
    );
  }
}

render(h(App), document.getElementById("root"));
