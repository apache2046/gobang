const { h, render, Component } = require("preact");
const { useState, useEffect } = require('preact/hooks');

const { Goban } = require("@sabaki/shudan");
const axios = require('axios').default

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
      showSelectMenu: true,
      player:1,
      waiting:false,
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
      showSelectMenu,
      player,
      waiting
    } = this.state;

    console.log('in render')


    if (!init)
      this.initBoard();

    console.log('in render2')
    return h(
      "div",
      {},
      h(
        "div",
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
          {
            style: {
              display: 'flex',
              flexDirection: 'row',
            }
          },
          h(
            'div',
            {
              style: {
                width: '15em',
                display: 'flex',
                flexDirection: 'column',
                paddingRight: '2em'
              }
            },
            h(
              'button',
              {},
              'haha1'
            ),
            h(
              'button',
              {},
              'haha2'
            )
          ),
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
              this.play(player, [x, y]).then(
                (data) => {
                  let [actor, pos, v_actor, patterns] = data
                  this.placeStone(actor, pos)
                  console.log(v_actor, patterns)
                }).then(() => {
                  return this.genmove(-player).then(
                    (data) => {
                      let [actor, pos, v_actor, patterns] = data
                      this.placeStone(actor, pos)
                      console.log(v_actor, patterns)
                    }
                  )
                }).then(() => { this.waiting = false })
            },
          }),
          h(
            'div',
            {
              style: {
                width: '15em',
                display: 'flex',
                flexDirection: 'column',
                marginLeft: '1em',
                color: '#ffffff',
                backgroundColor: '#000000'
              }
            },
            h('p',
              {},
              "hahaha2 22222 22222 2222 22 222 222 2222"),
            h('p',
              {},
              "hahaha")
          )
        ),
      ),
      h(
        'div',
        {
          style:
          {
            display: showSelectMenu ? "flex":"none",
            position: 'absolute',
            top: '0px',
            bottom: '0px',
            left: '0px',
            right: '0px',
            'z-index': 999,
            color: '#ffffff',
            'justify-content': 'center',
            // display: 'flex',
            backgroundColor: '#000000B0'
          },
        },

        // h('span', {
        //   style: {
        //     margin: 'auto'
        //   }
        // },
          h('form', {
            style: {
              display: 'flex',
              flexDirection: 'column',
              background: '#888888',
              padding: '1em',
              margin: 'auto',
              // "box-shadow": "10px 10px"
            }
          },
            h('span', {}, "请选择谁先："),
            h('button', {
              type: 'button',
              onClick: evt => {
                this.setState({ showSelectMenu: false, player: 1})
                console.log('AAA')
              },
              style: {
                width: '12em', padding: '1em', margin: '1em',
              }
            }, "我先手"),

            h('button', {
              type: 'button',
              onClick: evt => {
                this.setState({ showSelectMenu: false , player:-1})
                this.genmove(1).then(
                  (data) => {
                    let [actor, pos, v_actor, patterns] = data
                    this.placeStone(actor, pos)
                    console.log(v_actor, patterns)
                  })
                console.log('BBB')
              },
              style: {
                width: '12em', padding: '1em', margin: '1em',
              }
            }, "电脑先手")
          )
        // )
      )
    )
  }
}

render(h(App), document.getElementById("root"));
