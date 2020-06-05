#load libraries
library(shiny)
# library(shinydashboard)


ui <- fluidPage(
    titlePanel("Deep fakes"),
    mainPanel(
        h3("Please enter your name"),
        textInput("name", ""),
        h3("First sample"),
        h4("Sound 1"),
        tags$audio(src = "3.wav", type = " 'audio/wav", controls = NA),
        h4("Sound 2"),
        tags$audio(src = "3.wav", type = " 'audio/wav", controls = NA),
        h4("Sound 3"),
        tags$audio(src = "3.wav", type = " 'audio/wav", controls = NA),
        h4("How well does it sound"),
        sliderInput("item1", "", 0, 10, value = 5),
        actionButton("submit", "Submit"),
        textOutput("ty"),
        img(src = "test.png", width = 350, height = 350)
    )
)

server <- function(input, output){
    sliderValues <- reactive({
        data.frame(Name = c("Sound 1"),
                   Value = c(input$item1),
                   stringsAsFactors = F)})
    observeEvent(input$submit, {output$ty <- renderText({paste0("Thanks for answering the survey ", input$name, "!")})})
}


shinyApp(ui = ui, server = server)