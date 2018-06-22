# load the required packages
library(shiny)
require(shinydashboard)
library(ggplot2)
library(dplyr)
library(sqldf)
require(stats)
require(lattice)

db <- dbConnect(SQLite(), dbname="~/Data/CADFinance.db")

#####
# getIntegrated
#
# returns a dataframe which contains all the information:
# date
# timeseries
# smoothedTS
# annualized return
# p-e
# div yield
getAll <- function(symb, smooth, force, close, curVal){
  print("GetAll()")
  
  sqlQuery <- paste("SELECT DISTINCT day_, month_, year_, close, 0 as wTran, 0 as annualized, eps, div FROM (SELECT day_, month_, year_ FROM xtse WHERE symbol = '", symb, "' UNION all SELECT 15 as day_, month_, year_ FROM earnings WHERE symbol = '", symb, "') LEFT JOIN( SELECT 15 as day2, month_ as month2, year_ as year2, div, eps FROM earnings WHERE symbol = '", symb, "') ON day_ = day2 AND month_ = month2 AND year_ = year2 LEFT JOIN ( SELECT day_ as day3, month_ as month3, year_ as year3, close FROM xtse WHERE symbol = '", symb, "') ON day_ = day3 AND month_ = month3 AND year_ = year3 ORDER BY year_, month_, day_;", sep = "")
  print(sqlQuery)
  rs <- dbSendQuery(db, sqlQuery) 
  
  while (!dbHasCompleted(rs)) {
    values <- dbFetch(rs)
    #print(dbFetch(rs))
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
  kern <- c(0,0.000001,0.000002,0.000005,0.000012,0.000027,0.00006,0.000125,0.000251,0.000484,0.000898,0.001601,0.002743,0.004514,0.00714,0.010852,0.015849,0.022242,0.029993,0.038866,0.048394,0.057904,0.066574,0.073551,0.078084,0.079656,0.078084,0.073551,0.066574,0.057904,0.048394,0.038866,0.029993,0.022242,0.015849,0.010852,0.00714,0.004514,0.002743,0.001601,0.000898,0.000484,0.000251,0.000125,0.00006,0.000027,0.000012,0.000005,0.000002,0.000001,0)
  wTrans <- convolve(ret[["close"]], kern, type = "filter")
  
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
  
  #Close only
  sqlQuery2 <- paste("SELECT day_, month_, year_, close, 0 as wTran, 0 as ann, 0 as eps, 0 as div FROM xtse WHERE symbol = '", symb, "';", sep = "")
  rs2 <- dbSendQuery(db, sqlQuery2) 
  while (!dbHasCompleted(rs2)) {
    finalVals <- dbFetch(rs2)
  }
  
  today <- tail(ret, 1)
  
  #fill in the annualized, p-e, and div from the sparse data
  j <- 1
  for(i in 1:nrow(ret)){
    if(!(is.na(ret[[i, "close"]]))){
      #finalVals[j, "ann"] <- ret[i, "annualized"]  
      fYear <- today[["year_"]]
      fMon <- today[["month_"]]
      fDay <- today[["day_"]]
      fVal <- tail(ret[["close"]],1)
      
      if (force == TRUE) {
        fVal <<- curVal
      } else if (smooth == TRUE){
        fVal <<- tail(wTrans, 1)
      } 
      finalVals[j, "pe"] <- ret[i, "pe"]
      finalVals[j, "div"] <- ret[i, "divYld"]
 
      
      iYear <- finalVals[j, "year_"]
      iMon <- finalVals[j, "month_"]
      iDay <- finalVals[j, "day_"]
      iVal <- finalVals[j, "close"]
      finalVals[j, "ann"] <- getAnnualizedReturn(iDay, iMon, iYear, iVal, fDay, fMon, fYear, fVal)
      j <- j + 1
    }
    
  }
  
  #print(finalVals)
  
  #TODO: smoothed time series of same length as data frame
  #generate the kernel from here for now:
  #http://dev.theomader.com/gaussian-kernel-calculator/
  
  #kern = c(0,0.000001,0.000002,0.000005,0.000012,0.000027,0.00006,0.000125,0.000251,0.000484,0.000898,0.001601,0.002743,0.004514,0.00714,0.010852,0.015849,0.022242,0.029993,0.038866,0.048394,0.057904,0.066574,0.073551,0.078084,0.079656,0.078084,0.073551,0.066574,0.057904,0.048394,0.038866,0.029993,0.022242,0.015849,0.010852,0.00714,0.004514,0.002743,0.001601,0.000898,0.000484,0.000251,0.000125,0.00006,0.000027,0.000012,0.000005,0.000002,0.000001,0)
  #wTran <- convolve(finlaVals[["close"]], kern, type = "filter")
  
  #populate the annualized return
  #x[[i, "annualized"]] <-getAnnualizedReturn(x[[i, "day_"]], x[[i, month_]], x[[i, year]], x[[i, "close"]], finalDay, finalMonth, finalYear, finalVal)
  
  return(finalVals)
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

runInt <- function(symb){
  intData <- getIntegrated(symb)
  intData
}


# Define server logic required to draw a histogram
shinyServer(function(input, output) {
  peProj <- 0.0
  divProj <- 0.0
    
  print("Starting server")

  updateSymbol<-reactive({
    if(exists(input$symInp))
      input$symInp
    else
      "BNS"
  })
  
  #TODO: today's value must be passed to getAll() as a parameter (not used in calculating current value)
  observeEvent(input$update_button, {
    
    #update data
    print(paste("New symbol ", input$symInp))
    sym <- input$symInp
    
    currentVal <- 0.0
    force <- FALSE
    smooth <- FALSE
    close <- FALSE
    
    if(input$rb == "smooth"){
      #currentVal <<- wTran[nrow(wTran)]
      smooth <- TRUE
      print("Smooth selected")
    } else if (input$rb == "force"){
      #currentVal <<- input$priceText
      force <- TRUE
      print(paste("Forced value selected - ", input$priceText, sep = ""))
    } else {
      #currentVal <<- dataSer[[nrow(dataSer), "close"]] 
      close <- TRUE
      print("Last close selected")
    }    
    
    dataSer <- getAll(sym, smooth, force, close, currentVal)
    kern = c(0,0.000001,0.000002,0.000005,0.000012,0.000027,0.00006,0.000125,0.000251,0.000484,0.000898,0.001601,0.002743,0.004514,0.00714,0.010852,0.015849,0.022242,0.029993,0.038866,0.048394,0.057904,0.066574,0.073551,0.078084,0.079656,0.078084,0.073551,0.066574,0.057904,0.048394,0.038866,0.029993,0.022242,0.015849,0.010852,0.00714,0.004514,0.002743,0.001601,0.000898,0.000484,0.000251,0.000125,0.00006,0.000027,0.000012,0.000005,0.000002,0.000001,0)
    wTran <- convolve(dataSer[["close"]], kern, type = "filter")
    print(paste("New symbol ", sym))
    output$pricePlot <- renderPlot({
      #dataSer <- getIntegrated(sym)
      
      #ts <- runTS(sym)
      #get the time series
      #input$update_button
      ts <- dataSer[["close"]]
      
      # draw the raw data series
      if(!is.null(ts)){
        plot(ts, pch = ".")
      }
    })

    todaysDiv <- dataSer[[nrow(dataSer), "div"]] #tail(dataSer[["div"]], 1)
    todaysPE <- dataSer[[nrow(dataSer), "pe"]] #tail(dataSer[["pe"]], 1)
    
    print(paste("Today's div: ", todaysDiv, sep = ""))
    print(paste("Today's pe: ", todaysPE, sep = ""))
    print(paste("Today's price:", currentVal, sep = ""))
    
    output$smoothPlot <- renderPlot({
      kern <- c(0,0.000001,0.000002,0.000005,0.000012,0.000027,0.00006,0.000125,0.000251,0.000484,0.000898,0.001601,0.002743,0.004514,0.00714,0.010852,0.015849,0.022242,0.029993,0.038866,0.048394,0.057904,0.066574,0.073551,0.078084,0.079656,0.078084,0.073551,0.066574,0.057904,0.048394,0.038866,0.029993,0.022242,0.015849,0.010852,0.00714,0.004514,0.002743,0.001601,0.000898,0.000484,0.000251,0.000125,0.00006,0.000027,0.000012,0.000005,0.000002,0.000001,0)
      wTran <- convolve(dataSer[["close"]], kern, type = "filter")
      
      #wTran <- runTrans("BNS")
      if(!is.null(wTran)){
        plot(wTran, pch = ".")
      }
    })
    
    output$retPlot <- renderPlot({
      #retSer <- runRets(sym)
      if(!is.null(dataSer[["ann"]])){
        plot(dataSer[["ann"]], pch = ".")
      }
    })
    
    output$divPlot <- renderPlot({
      if(!is.null(dataSer[["div"]])){
        todaysDiv <<- dataSer[[nrow(dataSer), "div"]] #tail(dataSer[["div"]], 1)
        todaysPE <<- dataSer[[nrow(dataSer), "pe"]] #tail(dataSer[["pe"]], 1)
                #plot(dataSer[["divYld"]], pch = ".")
        dv.mod2 <- lm(ann ~ div, data = dataSer)
        summary(dv.mod2)
        plot(dataSer$div, dataSer$ann, xlab = "Dividend Yield", ylab = "Annualized Return")
        abline(dv.mod2)
        divProj <<- predict(dv.mod2, data.frame(div = c(todaysDiv)))
      }

    })
    

    
    output$pePlot <- renderPlot({
      if(!is.null(dataSer[["pe"]])){
        todaysDiv <<- dataSer[[nrow(dataSer), "div"]] #tail(dataSer[["div"]], 1)
        todaysPE <<- dataSer[[nrow(dataSer), "pe"]] #tail(dataSer[["pe"]], 1)
        #plot(dataSer[["pe"]], pch = ".")
        dv.mod1 <- lm(ann ~ pe, data = dataSer)
        summary(dv.mod1)
        plot(dataSer$pe, dataSer$ann, xlab = "P-E", ylab = "Annualized Return")
        abline(dv.mod1)
        peProj <<- predict(dv.mod1, data.frame(pe = c(todaysPE)))
      }
    })
    
    print(paste("Today's div: ", todaysDiv, sep = ""))
    print(paste("Today's pe: ", todaysPE, sep = ""))
    print(paste("PE PROJ: ", peProj, sep = ""))
    print(paste("DIV PROJ: ", divProj, sep = ""))

    output$divProjBox <- renderValueBox({
      valueBox(
        divProj, "Dividend Projection", icon = icon("list"),
        color = "purple"
      )
    })
    
    output$peProjBox <- renderValueBox({
      valueBox(
        peProj, "P-E Projection", icon = icon("list"),
        color = "yellow"
      )
    })
    
    
  })
  
  loadData <- function() {
    if (exists(input$symInp)) {
      input$symInp
    }
  }
  
  dataSer <- getAll("BNS", FALSE, TRUE, FALSE, 0.0)
  
  print(dataSer)
  print(dataSer[nrow(dataSer) -1,])
  print(dataSer[nrow(dataSer),])
  
  todaysDiv <- dataSer[[nrow(dataSer), "div"]] #tail(dataSer[["div"]], 1)
  todaysPE <- dataSer[[nrow(dataSer), "pe"]] #tail(dataSer[["pe"]], 1)
   
  output$pricePlot <- renderPlot({

    if(!is.null(dataSer)){
      plot(dataSer[["close"]], pch = ".")
    }
  })
  
  output$smoothPlot <- renderPlot({
    kern <- c(0,0.000001,0.000002,0.000005,0.000012,0.000027,0.00006,0.000125,0.000251,0.000484,0.000898,0.001601,0.002743,0.004514,0.00714,0.010852,0.015849,0.022242,0.029993,0.038866,0.048394,0.057904,0.066574,0.073551,0.078084,0.079656,0.078084,0.073551,0.066574,0.057904,0.048394,0.038866,0.029993,0.022242,0.015849,0.010852,0.00714,0.004514,0.002743,0.001601,0.000898,0.000484,0.000251,0.000125,0.00006,0.000027,0.000012,0.000005,0.000002,0.000001,0)
    wTran <- convolve(dataSer[["close"]], kern, type = "filter")
    
    #wTran <- runTrans("BNS")
    if(!is.null(wTran)){
      plot(wTran, pch = ".")
    }
  })
  
  output$retPlot <- renderPlot({
    #retSer <- runRets("BNS")
    if(!is.null(dataSer[["ann"]])){
      plot(dataSer[["ann"]], pch = ".")
    }
  })
  
  output$divPlot <- renderPlot({
    if(!is.null(dataSer[["div"]])){
      todaysDiv <<- dataSer[[nrow(dataSer), "div"]] #tail(dataSer[["div"]], 1)
      todaysPE <<- dataSer[[nrow(dataSer), "pe"]] #tail(dataSer[["pe"]], 1)
      #plot(dataSer[["divYld"]], pch = ".")
      dv.mod2 <- lm(ann ~ div, data = dataSer)
      summary(dv.mod2)
      plot(dataSer$div, dataSer$ann, xlab = "Dividend Yield", ylab = "Annualized Return")
      abline(dv.mod2)
      
      divProj <<- predict(dv.mod2, data.frame(div = c(todaysDiv)))
    }
  })
  
  output$pePlot <- renderPlot({
    todaysDiv <<- dataSer[[nrow(dataSer), "div"]] #tail(dataSer[["div"]], 1)
    todaysPE <<- dataSer[[nrow(dataSer), "pe"]] #tail(dataSer[["pe"]], 1)
    if(!is.null(dataSer[["pe"]])){
      #plot(dataSer[["pe"]], pch = ".")
      dv.mod1 <- lm(ann ~ pe, data = dataSer)
      summary(dv.mod1)
      plot(dataSer$pe, dataSer$ann, xlab = "P-E", ylab = "Annualized Return")
      abline(dv.mod1)
      peProj <<- predict(dv.mod1, data.frame(pe = c(todaysPE)))
    }
  })
  
  print(paste("Today's div: ", todaysDiv, sep = ""))
  print(paste("Today's pe: ", todaysPE, sep = ""))
  print(paste("DIV PROJ: ", divProj, sep = ""))
  print(paste("PE PROJ: ", peProj, sep = ""))
  
  output$divProjBox <- renderValueBox({
    valueBox(
      divProj, "Dividend Projection", icon = icon("list"),
      color = "purple"
    )
  })
  
  output$peProjBox <- renderValueBox({
    valueBox(
      peProj, "P-E Projection", icon = icon("list"),
      color = "yellow" 
    )
  })
  
})

#getIntegrated <- function(symb){
#  sqlQuery <- paste("SELECT DISTINCT day_, month_, year_, close, eps, div FROM (SELECT day_, month_, year_ FROM xtse WHERE symbol = '", symb, "' UNION all SELECT 15 as day_, month_, year_ FROM earnings WHERE symbol = '", symb, "') LEFT JOIN( SELECT 15 as day2, month_ as month2, year_ as year2, div, eps FROM earnings WHERE symbol = '", symb, "') ON day_ = day2 AND month_ = month2 AND year_ = year2 LEFT JOIN ( SELECT day_ as day3, month_ as month3, year_ as year3, close FROM xtse WHERE symbol = '", symb, "') ON day_ = day3 AND month_ = month3 AND year_ = year3 ORDER BY year_, month_, day_;", sep = "")
#  print(sqlQuery)
#  rs <- dbSendQuery(db, sqlQuery) 
  
#  while (!dbHasCompleted(rs)) {
#    values <- dbFetch(rs)
#    print(dbFetch(rs))
#  }
  
#  ret <- values
  
  #counts the number of valid data points we have for earnings
#  earnsCount <- 0
  
  #track the last four eps and div
#  earns <- c(0,0,0,0)
#  divs <- c(0,0,0,0)
#  anEPS <- 0.0
#  anDiv <- 0.0
  
#  for(i in 1:nrow(ret)){
#    if(!is.na(ret[[i, "eps"]])){
#      earnsCount <- earnsCount + 1
#      earns[1] <- earns[2]
#      earns[2] <- earns[3]
#      earns[3] <- earns[4]
#      earns[4] <- ret[i, "eps"]
#      anEPS <- earns[1] + earns[2] + earns[3] + earns[4]
      
#      divs[1] <- divs[2]
#      divs[2] <- divs[3]
#      divs[3] <- divs[4]
#      divs[4] <- ret[i, "div"]
#      anDiv <- divs[1] + divs[2] + divs[3] + divs[4]
      
      #print(paste("EPS: ", anEPS, " Div: ", anDiv, sep = ""))
#    }
    
#    if(!is.na(ret[i, "close"]) && earnsCount >= 4){
#      ret[i, "pe"] <- ret[i, "close"] / anEPS
#      ret[i, "divYld"] <- 100 * anDiv / ret[i, "close"] 
#    }
#  }
  
#  return(ret)
  
#}
