#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)

# Define UI for application that draws a histogram
shinyUI(fluidPage(
  
  # Application title
  titlePanel("Financial Data"),

  # Textinput selector for the symbol 
  sidebarLayout(
    sidebarPanel(
      textInput(inputId = "symInp", label = "Symbol", value = ""),
      actionButton("update_button", "Update chart")
    ),
    
    # Plots of raw and smoothed data
    mainPanel(
       plotOutput("pricePlot"),
       plotOutput("smoothPlot")
    )
  )
))
