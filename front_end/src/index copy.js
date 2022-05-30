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
    // const [bdata, setBdata] = useState({size:11 });
    // const [bdata, setBdata] = useState(this.state);
    console.log('in render')
    useEffect(() => {
      console.log('in effect')
      const fetchData = async () => {
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
  
        // setBdata({size:11 });
        // setBdata({...bdata, 'isBusy':false})
        this.setState({'isBusy':false})
        console.log('end axio')
      };
  
      fetchData();
      this.state.isBusy = false
      console.log('end effect')


    }, []);
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

          vertexSize: this.state.vertexSize,
          animate: true,
          busy: this.state.isBusy,
          coordX: this.state.alternateCoordinates ? (i) => chineseCoord[i] : undefined,
          coordY: this.state.alternateCoordinates ? (i) => i + 1 : undefined,

          signMap: this.state.board,
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
          },
        })
      )
    );
  }
}

render(h(App), document.getElementById("root"));
