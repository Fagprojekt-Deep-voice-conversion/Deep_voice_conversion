#load libraries
library(shiny)
# library(shinydashboard)


ui <- fluidPage(
    titlePanel("Deep fakes"),
    mainPanel(
        h3("Please enter your name"),
        textInput("name", ""),
        # h3("First sample"),
        # h4("Sound 1"),
        # tags$audio(src = "3.wav", type = " 'audio/wav", controls = NA),
        # h4("Sound 2"),
        # tags$audio(src = "3.wav", type = " 'audio/wav", controls = NA),
        # h4("Sound 3"),
        # tags$audio(src = "3.wav", type = " 'audio/wav", controls = NA),
        h3("First sample"),
        actionButton("play1", "Play sound 1"),
        actionButton("play2", "Play sound 2"),
        actionButton("play3", "Play sound 3"),
        # h4("How well does it sound"),
        sliderInput("item1", "How well does it sound?", 0, 10, value = 5),
        textOutput("score1"),
        actionButton("submit", "Submit"),
        textOutput("ty"),
        actionButton("scared", "Get haunted!")
        
        
    )
)

server <- function(input, output){
    # sliderValues <- reactive({
    #     data.frame(Name = c("Sound 1"),
    #                Value = c(input$item1),
    #                stringsAsFactors = F)})
    observeEvent(input$play1, {insertUI(ui = tags$audio(src = "1.wav", type = " 'audio/wav", autoplay = NA, controls = NA, style="display:none;"), selector = "#play1", where = "afterEnd")})
    observeEvent(input$play2, {insertUI(ui = tags$audio(src = "2.wav", type = " 'audio/wav", autoplay = NA, controls = NA, style="display:none;"), selector = "#play2", where = "afterEnd")})
    observeEvent(input$play3, {insertUI(ui = tags$audio(src = "3.wav", type = " 'audio/wav", autoplay = NA, controls = NA, style="display:none;"), selector = "#play3", where = "afterEnd")})
    
    observeEvent(input$item1, {output$score1 <- renderText({paste0("You gave a score of ", input$item1)})})
    
    observeEvent(input$submit, {output$ty <- renderText({paste0("Thanks for answering the survey ", input$name, "!")})})
    observeEvent(input$scared, {insertUI(ui = img(src = "test.png", width = 350, height = 350), selector = "#scared", where = "afterEnd")})
    
}


shinyApp(ui = ui, server = server)