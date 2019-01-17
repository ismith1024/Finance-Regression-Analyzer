# Finance Regression Learner

Personal finance project - predicts five year returns given fundamental analysis parameters.


### Table of Contents

1. [Libraries and Installation](#installation)
2. [Project Description](#motivation)
3. [Files](#files)
4. [TODO](#results)

## Libraries and Installation <a name="installation"></a>

This project is written in the R language, using the RStudio IDE.
The web app uses the Shiny framework: https://shiny.rstudio.com/tutorial/


## Project Description<a name="motivation"></a>

The purpose of this project is to predict five-year market returns for equities with "predictable" growth.
Using fundamental analysis, the purpose is to predict long-term returns using a Value investing style.  This
is in contrast to most trading bots, which perform technical analysis on the time series price values.

## Files <a name="files"></a>

server.r - contains the core functionality of the app
ui.r - renders the HTML
TODO: 
project depends on a SQLite database "~/Data/CADFinance.db".  Not part of this repository at this time.

## Notes<a name="results"></a>

This application provides real-world useful analysis, but the UI should probably be improved.
I would like to improve the gaussian kernal functions to use a long kernel for current price and a short kernel for historic price data.  Kernel functions need to be written to accommodate one-sided kernels.

The project is still active and I will improve it once time permits.