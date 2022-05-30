const path = require("path");

const HtmlWebpackPlugin = require("html-webpack-plugin");
const { CleanWebpackPlugin } = require("clean-webpack-plugin");
const CopyWebpackPlugin = require("copy-webpack-plugin");

module.exports = {
  mode: "development",
  devtool: 'inline-source-map',
  entry: {
    main: "./src/index.js",
  },
  output: {
    filename: "bundle.js",
    path: path.resolve(__dirname, "dist"),
  },
  // module: {
  //   rules: [
  //     {
  //       test: /.(jpg|png|gif)$/,
  //       use: {
  //         loader: "file-loader",
  //         options: {
  //           // 占位符 placeholder
  //           name: "[name]_[hash].[ext]",
  //           outputPath: "images/",
  //         },
  //       },
  //     },
  //   ],
  // },
  plugins: [
    new HtmlWebpackPlugin({
      template: "./src/index.html",
    }),
    new CleanWebpackPlugin(),
    new CopyWebpackPlugin({
      patterns: [
        { from: "src/css", to: "css" },
        { from: "src/imgs", to: "imgs" },
        { from: "src/audio", to: "audio" },
      ],
    }),
  ],
};
