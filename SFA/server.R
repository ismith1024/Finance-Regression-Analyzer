library(shiny)
library(sqldf)
require(stats)
require(lattice)

db <- dbConnect(SQLite(), dbname="~/Data/CADFinance.db")

#####
# getAll
#
# returns a dataframe which contains all the information:
# date
# timeseries
# smoothedTS
# annualized return
# p-e
# div yield

#TODO: count the non-null close dates an put into proper data frame (asfer counting the PE and DIV)
getAll <- function(symb){
  print("GetAll()")

  sqlQuery <- paste("SELECT DISTINCT day_, month_, year_, close, 0 as wTran, 0 as annualized, eps, div FROM (SELECT day_, month_, year_ FROM xtse WHERE symbol = '", symb, "' UNION all SELECT 15 as day_, month_, year_ FROM earnings WHERE symbol = '", symb, "') LEFT JOIN( SELECT 15 as day2, month_ as month2, year_ as year2, div, eps FROM earnings WHERE symbol = '", symb, "') ON day_ = day2 AND month_ = month2 AND year_ = year2 LEFT JOIN ( SELECT day_ as day3, month_ as month3, year_ as year3, close FROM xtse WHERE symbol = '", symb, "') ON day_ = day3 AND month_ = month3 AND year_ = year3 ORDER BY year_, month_, day_;", sep = "")
  print(sqlQuery)
  rs <- dbSendQuery(db, sqlQuery) 
  
  while (!dbHasCompleted(rs)) {
    values <- dbFetch(rs)
    print(dbFetch(rs))
  }
  
  ret <- values
  
  #counts the number of valid data points we have for earnings
  earnsCount <- 0
  
  #track the last four eps and div
  earns <- c(0,0,0,0)
  divs <- c(0,0,0,0)
  anEPS <- 0.0
  anDiv <- 0.0

  #smooth the data series
  #kern <- c(0,0.000001,0.000002,0.000005,0.000012,0.000027,0.00006,0.000125,0.000251,0.000484,0.000898,0.001601,0.002743,0.004514,0.00714,0.010852,0.015849,0.022242,0.029993,0.038866,0.048394,0.057904,0.066574,0.073551,0.078084,0.079656,0.078084,0.073551,0.066574,0.057904,0.048394,0.038866,0.029993,0.022242,0.015849,0.010852,0.00714,0.004514,0.002743,0.001601,0.000898,0.000484,0.000251,0.000125,0.00006,0.000027,0.000012,0.000005,0.000002,0.000001,0)
  #ret[["wTrans"]] <- convolve(ret[["close"]], kern, type = "filter")
  
  fYear <- today[["year_"]]
  fMon <- today[["month_"]]
  fDay <- today[["day_"]]
  fVal <- tail(ret[["wTrans"]],1)
    
  for(i in 1:nrow(ret)){

    if(!is.na(ret[[i, "eps"]])){
      earnsCount <- earnsCount + 1
      earns[1] <- earns[2]
      earns[2] <- earns[3]
      earns[3] <- earns[4]
      earns[4] <- ret[i, "eps"]
      anEPS <- earns[1] + earns[2] + earns[3] + earns[4]
      
      divs[1] <- divs[2]
      divs[2] <- divs[3]
      divs[3] <- divs[4]
      divs[4] <- ret[i, "div"]
      anDiv <- divs[1] + divs[2] + divs[3] + divs[4]
      
      #print(paste("EPS: ", anEPS, " Div: ", anDiv, sep = ""))
    }
    
    if(!is.na(ret[i, "close"]) && earnsCount >= 4){
      ret[i, "pe"] <- ret[i, "close"] / anEPS
      ret[i, "divYld"] <- 100 * anDiv / ret[i, "close"] 
    }
  }
  
  #count the non-null close values
  closeCount <- 0
  for(i in 1:nrow(ret)){
    if(!(is.na(ret[i, "close"]))){
      closeCount <- closeCount + 1
    }
  }
  
  #print(ret)
  
  #Close only
  sqlQuery2 <- paste("SELECT day_, month_, year_, close, 0 as wTran, 0 as ann, 0 as eps, 0 as div FROM xtse WHERE symbol = '", symb, "';", sep = "")
  rs2 <- dbSendQuery(db, sqlQuery2) 
  while (!dbHasCompleted(rs2)) {
    x <- dbFetch(rs2)
  }
  
  #fill in the annualized, p-e, and div from the sparse data
  j <- 0
  for(i in 1:nrow(ret)){
    if(!(is.na(ret[[i, "close"]]))){
      x[j, "ann"] <- ret[i, "annualized"]
      x[j, "pe"] <- ret[i, "pe"]
      x[j, "div"] <- ret[i, "divYld"]
      j <- j + 1
    }
  }

  #TODO: this is still a mess
  #need to put the filtered time series into the middle of the data frame
  #smooth the time series
  #generate the kernel from here for now:
  #http://dev.theomader.com/gaussian-kernel-calculator/
  
  kern = c(0,0.000001,0.000002,0.000005,0.000012,0.000027,0.00006,0.000125,0.000251,0.000484,0.000898,0.001601,0.002743,0.004514,0.00714,0.010852,0.015849,0.022242,0.029993,0.038866,0.048394,0.057904,0.066574,0.073551,0.078084,0.079656,0.078084,0.073551,0.066574,0.057904,0.048394,0.038866,0.029993,0.022242,0.015849,0.010852,0.00714,0.004514,0.002743,0.001601,0.000898,0.000484,0.000251,0.000125,0.00006,0.000027,0.000012,0.000005,0.000002,0.000001,0)
  xx <- convolve(x[["close"]], kern, type = "filter")
  
  print("X:")
  print(x)
  print(nrow(x))
  print(length(xx))
  
  #populate the annualized return
  #x[[i, "annualized"]] <-getAnnualizedReturn(x[[i, "day_"]], x[[i, month_]], x[[i, year]], x[[i, "close"]], finalDay, finalMonth, finalYear, finalVal)
  
  
  
  #summary(dv.mod1)
  #dv.mod1 <- lm(divYld ~ pe, data = ret)
  #plot(ret$divYld, ret$pe, main = paste("Demo Regression: ", symb, sep = ""), xlab = "Dividend Yield", ylab = "P-E")
  
  #abline(dv.mod1)
  
  
  return(ret)
}


#get the time series
getTimeSeries <- function(symb){
  sqlQuery <- paste("SELECT close, day_, month_, year_ FROM xtse WHERE symbol = '", symb, "';", sep = "")
  #print(sqlQuery)
  rs <- dbSendQuery(db, sqlQuery) #time series
  while (!dbHasCompleted(rs)) {
    values <- dbFetch(rs)
  }
  
  timeSeries <- values
  dbClearResult(rs)
  return(timeSeries)

}

# provides the annualized return of the equity to the most recent date on record
# (today's value / initial value)^(1/time); time in years
# ser is the time series of values
getAnnualizedReturn <- function(initDay, initMonth, initYear, initVal, finalDay, finalMonth, finalYear, finalVal){
  years <- finalYear - initYear
  years <- years + (finalMonth - initMonth) / 12
  years <- years + (finalDay - initDay) / 252
  
  change <- (finalVal/initVal)^(1/years) 
  return(change)
}

#get time series and compute the return data for a symbol
#time series is smoothed using a Weiersrass transform -- intended to remove Gaussian noise
#Kernel prameters:
#n = 51
#sigma = 10
runRets <- function(symb){
  timeSer <- getTimeSeries(symb)
  #generate the kernel from here for now:
  #http://dev.theomader.com/gaussian-kernel-calculator/
  kern = c(0,0.000001,0.000002,0.000005,0.000012,0.000027,0.00006,0.000125,0.000251,0.000484,0.000898,0.001601,0.002743,0.004514,0.00714,0.010852,0.015849,0.022242,0.029993,0.038866,0.048394,0.057904,0.066574,0.073551,0.078084,0.079656,0.078084,0.073551,0.066574,0.057904,0.048394,0.038866,0.029993,0.022242,0.015849,0.010852,0.00714,0.004514,0.002743,0.001601,0.000898,0.000484,0.000251,0.000125,0.00006,0.000027,0.000012,0.000005,0.000002,0.000001,0)
  wTrans = convolve(timeSer[["close"]], kern, type = "filter")
  #plot(wTrans, pch = ".")
  today <- tail(timeSer, 1)
  
  fYear <- today[["year_"]]
  fMon <- today[["month_"]]
  fDay <- today[["day_"]]
  fVal <- tail(wTrans,1)
  rets <- vector("numeric", nrow(timeSer))
  
  #for(i in 1:10){
  for(i in 1:nrow(timeSer)){
    iYear <- timeSer[i, "year_"]
    iMon <- timeSer[i, "month_"]
    iDay <- timeSer[i, "day_"]
    iVal <- wTrans[i]
    rets[i] <- getAnnualizedReturn(iDay, iMon, iYear, iVal, fDay, fMon, fYear, fVal)
    
  }
  
  print(today)
  plot(rets, pch = ".")

}

runTS <- function(symb){
  print(paste("Run TS: ", symb))
  timeSer <- getTimeSeries(symb)
  plot(timeSer[["close"]], pch = ".")
  return(timeSer[["close"]])
}

runTrans <- function(symb){
  timeSer <- getTimeSeries(symb)
  #generate the kernel from here for now:
  #http://dev.theomader.com/gaussian-kernel-calculator/
  kern = c(0,0.000001,0.000002,0.000005,0.000012,0.000027,0.00006,0.000125,0.000251,0.000484,0.000898,0.001601,0.002743,0.004514,0.00714,0.010852,0.015849,0.022242,0.029993,0.038866,0.048394,0.057904,0.066574,0.073551,0.078084,0.079656,0.078084,0.073551,0.066574,0.057904,0.048394,0.038866,0.029993,0.022242,0.015849,0.010852,0.00714,0.004514,0.002743,0.001601,0.000898,0.000484,0.000251,0.000125,0.00006,0.000027,0.000012,0.000005,0.000002,0.000001,0)
  wTrans = convolve(timeSer[["close"]], kern, type = "filter")
  plot(wTrans, pch = ".")
  return(wTrans)
}

getIntegrated <- function(symb){
  sqlQuery <- paste("SELECT DISTINCT day_, month_, year_, close, eps, div FROM (SELECT day_, month_, year_ FROM xtse WHERE symbol = '", symb, "' UNION all SELECT 15 as day_, month_, year_ FROM earnings WHERE symbol = '", symb, "') LEFT JOIN( SELECT 15 as day2, month_ as month2, year_ as year2, div, eps FROM earnings WHERE symbol = '", symb, "') ON day_ = day2 AND month_ = month2 AND year_ = year2 LEFT JOIN ( SELECT day_ as day3, month_ as month3, year_ as year3, close FROM xtse WHERE symbol = '", symb, "') ON day_ = day3 AND month_ = month3 AND year_ = year3 ORDER BY year_, month_, day_;", sep = "")
  print(sqlQuery)
  rs <- dbSendQuery(db, sqlQuery) 
  
  while (!dbHasCompleted(rs)) {
    values <- dbFetch(rs)
    print(dbFetch(rs))
  }
  
  ret <- values
  
  #counts the number of valid data points we have for earnings
  earnsCount <- 0
  
  #track the last four eps and div
  earns <- c(0,0,0,0)
  divs <- c(0,0,0,0)
  anEPS <- 0.0
  anDiv <- 0.0
  
  for(i in 1:nrow(ret)){
    if(!is.na(ret[[i, "eps"]])){
      earnsCount <- earnsCount + 1
      earns[1] <- earns[2]
      earns[2] <- earns[3]
      earns[3] <- earns[4]
      earns[4] <- ret[i, "eps"]
      anEPS <- earns[1] + earns[2] + earns[3] + earns[4]
      
      divs[1] <- divs[2]
      divs[2] <- divs[3]
      divs[3] <- divs[4]
      divs[4] <- ret[i, "div"]
      anDiv <- divs[1] + divs[2] + divs[3] + divs[4]
      
      #print(paste("EPS: ", anEPS, " Div: ", anDiv, sep = ""))
    }
    
    if(!is.na(ret[i, "close"]) && earnsCount >= 4){
      ret[i, "pe"] <- ret[i, "close"] / anEPS
      ret[i, "divYld"] <- 100 * anDiv / ret[i, "close"] 
    }
  }

  #summary(dv.mod1)
  dv.mod1 <- lm(divYld ~ pe, data = ret)
  plot(ret$divYld, ret$pe, main = paste("Demo Regression: ", symb, sep = ""), xlab = "Dividend Yield", ylab = "P-E")
  
  abline(dv.mod1)
  
  
  return(ret)
  
}

runInt <- function(symb){
  intData <- getIntegrated(symb)
  intData
}


# Define server logic required to draw a histogram
shinyServer(function(input, output) {
  
  print("Starting server")
  
  all <- getAll("BNS")
  
  print(all)
  
  updateSymbol<-reactive({
    if(exists(input$symInp))
      input$symInp
    else
      "BNS"
  })

  observeEvent(input$update_button, {
    #update data
    print(paste("New symbol ", input$symInp))
    sym <- input$symInp
    dataSer <- runInt(sym)
    print(paste("New symbol ", sym))
    output$pricePlot <- renderPlot({
      dataSer <- getIntegrated(sym)
      
        #ts <- runTS(sym)
        #get the time series
        #input$update_button
        ts <- dataSer[["close"]]
      
        # draw the raw data series
        if(!is.null(ts)){
          plot(ts, pch = ".")
        }
      })
      
      output$smoothPlot <- renderPlot({
        wTran <- runTrans(sym)
        #get the smoothed time series
        input$update_button
        
        # draw the smoothed data plot
        if(!is.null(wTran)){
          plot(wTran, pch = ".")
        }
      })
      
      output$retPlot <- renderPlot({
        retSer <- runRets(sym)
        if(!is.null(retSer)){
          plot(retSer, pch = ".")
        }
      })
      
      output$divPlot <- renderPlot({
        if(!is.null(dataSer[["divYld"]])){
          plot(dataSer[["divYld"]], pch = ".")
        }
      })
      
      output$pePlot <- renderPlot({
        if(!is.null(dataSer[["pe"]])){
          plot(dataSer[["pe"]], pch = ".")
        }
      })
      
  })
  
  loadData <- function() {
    if (exists(input$symInp)) {
      input$symInp
    }
  }
  
  dataSer <- getIntegrated("BNS")
  
  output$pricePlot <- renderPlot({
    ts <- runTS("BNS")
    if(!is.null(ts)){
      plot(ts, pch = ".")
    }
  })
  
  output$smoothPlot <- renderPlot({
    wTran <- runTrans("BNS")
    if(!is.null(wTran)){
      plot(wTran, pch = ".")
    }
  })
  
  output$retPlot <- renderPlot({
    retSer <- runRets("BNS")
    if(!is.null(retSer)){
      plot(retSer, pch = ".")
    }
  })
  
  output$divPlot <- renderPlot({
    if(!is.null(dataSer[["divYld"]])){
      plot(dataSer[["divYld"]], pch = ".")
    }
  })
  
  output$pePlot <- renderPlot({
    if(!is.null(dataSer[["pe"]])){
      plot(dataSer[["pe"]], pch = ".")
    }
  })
  
   
})



