# load the required packages
library(shiny)
require(shinydashboard)
library(ggplot2)
library(dplyr)
library(sqldf)
require(stats)
require(lattice)

#Dashboard header carrying the title of the dashboard

frow0 <- fluidRow(
  valueBoxOutput("divProjBox"),
  
  valueBoxOutput("peProjBox")
)

frow1 <- fluidRow(
  valueBoxOutput("value1")
  ,valueBoxOutput("value2")
  ,valueBoxOutput("value3")
)
frow2 <- fluidRow( 
  # time series
  box(
    title = "Time series"
    ,status = "primary"
    ,solidHeader = TRUE 
    ,collapsible = TRUE 
    ,plotOutput("pricePlot", height = "300px")
  ),
  #smoothed time series
  box(
    title = "Smoothed time series"
    ,status = "primary"
    ,solidHeader = TRUE 
    ,collapsible = TRUE 
    ,plotOutput("smoothPlot", height = "300px")
  ) 
)

frow3 <- fluidRow( 
  #return to Date
  box(
    title = "Returns to date"
    ,status = "primary"
    ,solidHeader = TRUE 
    ,collapsible = TRUE 
    ,plotOutput("retPlot", height = "300px")
  )
)

frow4 <- fluidRow( 
  #p-e regression
  box(
    title = "p-e - return Regression"
    ,status = "primary"
    ,solidHeader = TRUE 
    ,collapsible = TRUE 
    ,plotOutput("pePlot", height = "300px")
  ),
  #dividend regression
  box(
    title = "Div Yield - return Regression"
    ,status = "primary"
    ,solidHeader = TRUE 
    ,collapsible = TRUE 
    ,plotOutput("divPlot", height = "300px")
  ) 
)

header <- dashboardHeader(title = "Financial Analysis")  
#Sidebar content of the dashboard
sidebar <- dashboardSidebar(
  sidebarMenu(
    sidebarSearchForm(textId = "symInp", buttonId = "update_button", label = "Symbol")
    #menuItem("Dashboard", tabName = "dashboard", icon = icon("dashboard")),
    #menuItem("Visit-us", icon = icon("send",lib='glyphicon'), href = "https://www.salesforce.com")
    
  )
)


# combine the fluid rows to make the body
body <- dashboardBody(frow0, frow1, frow2, frow3, frow4)

ui <- dashboardPage(
  header,
  sidebar,
  body
)

