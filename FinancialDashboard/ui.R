# load the required packages
library(shiny)
require(shinydashboard)
library(ggplot2)
library(dplyr)
library(sqldf)
require(stats)
require(lattice)

# Define UI for application that draws a histogram
#shinyUI(fluidPage(
  
  # Application title
  #titlePanel("Old Faithful Geyser Data"),
  
  # Sidebar with a slider input for number of bins 
  #sidebarLayout(
  #  sidebarPanel(
  #     sliderInput("bins",
  #                 "Number of bins:",
  #                 min = 1,
  #                 max = 50,
  #                 value = 30)
  #  ),
    
    # Show a plot of the generated distribution
  #  mainPanel(
  #     plotOutput("distPlot")
  #  )ui <- dashboardPage(
  #)
#))

#Dashboard header carrying the title of the dashboard
header <- dashboardHeader(title = "Financial Analysis")  
#Sidebar content of the dashboard
sidebar <- dashboardSidebar(
  sidebarMenu(
    menuItem("Dashboard", tabName = "dashboard", icon = icon("dashboard")),
    menuItem("Visit-us", icon = icon("send",lib='glyphicon'), 
             href = "https://www.salesforce.com")
  )
)

frow0 <- fluidRow(
  textInput(inputId = "symInp", label = "Symbol", value = ""),
  actionButton("update_button", "Update chart")
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

# combine the two fluid rows to make the body
body <- dashboardBody(frow0, frow1, frow2, frow3, frow4)

ui <- dashboardPage(
  header,
  sidebar,
  body
)

