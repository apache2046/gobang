const { h, render, Component } = require("preact");
const { Goban } = require("@sabaki/shudan");

const signMap = [
  [0, 0, 0, -1, -1, -1, 1, 0, 1, 1, -1, -1, 0, -1, 0, -1, -1, 1, 0],
  [0, 0, -1, 0, -1, 1, 1, 1, 0, 1, -1, 0, -1, -1, -1, -1, 1, 1, 0],
  [0, 0, -1, -1, -1, 1, 1, 0, 0, 1, 1, -1, -1, 1, -1, 1, 0, 1, 0],
  [0, 0, 0, 0, -1, -1, 1, 0, 1, -1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
  [0, 0, 0, 0, -1, 0, -1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
  [0, 0, -1, 0, 0, -1, -1, 1, 0, -1, -1, 1, -1, -1, 0, 1, 0, 0, 1],
  [0, 0, 0, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1],
  [0, 0, -1, 1, 1, 0, 1, -1, -1, 1, 0, 1, -1, 0, 1, -1, -1, -1, 1],
  [0, 0, -1, -1, 1, 1, 1, 0, -1, 1, -1, -1, 0, -1, -1, 1, 1, 1, 1],
  [0, 0, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1],
  [-1, -1, -1, -1, 1, 1, 1, -1, 0, -1, 1, -1, -1, 0, -1, 1, 1, -1, 0],
  [-1, 1, -1, 0, -1, -1, -1, -1, -1, -1, 1, -1, 0, -1, -1, 1, -1, 0, -1],
  [1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 0, 1, -1, 0, -1, 1, -1, -1, 0],
  [0, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1],
  [0, 0, -1, 1, 0, 0, 1, 1, -1, -1, 0, 1, -1, 1, -1, 1, -1, 0, -1],
  [0, 0, 1, 0, 1, 0, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 0],
  [0, 0, 0, 0, 1, 1, 0, 1, -1, 0, -1, -1, 1, 1, 1, 1, -1, -1, -1],
  [0, 0, 1, 1, -1, 1, 1, -1, 0, -1, -1, 1, 1, 1, 1, 0, 1, -1, 1],
  [0, 0, 0, 1, -1, -1, -1, -1, -1, 0, -1, -1, 1, 1, 0, 1, 1, 1, 0],
];

class App extends Component {
  constructor(props) {
    super(props);

    this.state = {
      board: signMap,
      vertexSize: 24,
      showCoordinates: false,
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
      isBusy: false,
    };
  }

  render() {
    let {
      vertexSize,
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
    } = this.state;

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
          busy: this.state.isBusy,
          rangeX: showCorner ? [8, 18] : undefined,
          rangeY: showCorner ? [12, 18] : undefined,
          coordX: alternateCoordinates ? (i) => chineseCoord[i] : undefined,
          coordY: alternateCoordinates ? (i) => i + 1 : undefined,

          signMap: this.state.board,
          showCoordinates,
          fuzzyStonePlacement,
          animateStonePlacement,
          paintMap: showPaintMap && paintMap,
          heatMap: showHeatMap && heatMap,
          markerMap: showMarkerMap && markerMap,
          ghostStoneMap: showGhostStones && ghostStoneMap,

          onVertexMouseUp: (evt, [x, y]) => {
            let sign = evt.button === 0 ? 1 : -1;
            //   let newBoard = this.state.board.makeMove(sign, [x, y])

            //   this.setState({board: newBoard})
          },
        })
      )
    );
  }
}

render(h(App), document.getElementById("root"));
