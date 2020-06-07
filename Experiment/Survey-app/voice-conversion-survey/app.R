#load libraries
library(shiny)
# library(shinydashboard)


ui <- fluidPage(
    titlePanel("Deep fakes"),
    sidebarPanel(
        h3("Please enter your name"),
        textInput("name", ""),
        # h3("First sample"),
        # h4("Sound 1"),
        # tags$audio(src = "3.wav", type = " 'audio/wav", controls = NA),
        # h4("Sound 2"),
        # tags$audio(src = "3.wav", type = " 'audio/wav", controls = NA),
        # h4("Sound 3"),
        # tags$audio(src = "3.wav", type = " 'audio/wav", controls = NA),
        
        textOutput("ty"),
        h3("First sample"),
        actionButton("play1", "Play sound 1"),
        actionButton("play2", "Play sound 2"),
        actionButton("play3", "Play sound 3"),
        # h4("How well does it sound"),
        sliderInput("item1", "How well does it sound?", 0, 10, value = 5),
        textOutput("score1"),
        h3("Second sample"),
        actionButton("play1", "Play sound 1"),
        actionButton("play2", "Play sound 2"),
        actionButton("play3", "Play sound 3"),
        # h4("How well does it sound"),
        sliderInput("item2", "How well does it sound?", 0, 10, value = 5),
        actionButton("submit", "Submit"),
        textOutput("ty1")),
        #actionButton("scared", "Get haunted!")
        mainPanel(
            # Main Action is where most everything is happenning in the
            # object (where the welcome message, survey, and results appear)
            uiOutput("MainAction"),
            # This displays the action putton Next.
            actionButton("Click.Counter", "Next"), 
            actionButton("submit", "Submit")    
        )
        
        
        
    
)

server <- function(input, output){
    # sliderValues <- reactive({
    #     data.frame(Name = c("Sound 1"),
    #                Value = c(input$item1),
    #                stringsAsFactors = F)})

    
    #observeEvent(input$Click.Counter, {output$ty <- renderText({paste0("You gave a score of ",counter)})})
    observeEvent(input$play1, {insertUI(ui = tags$audio(src = "1.wav", type = " 'audio/wav", autoplay = NA, controls = NA, style="display:none;"), selector = "#play1", where = "afterEnd")})
    observeEvent(input$play2, {insertUI(ui = tags$audio(src = "2.wav", type = " 'audio/wav", autoplay = NA, controls = NA, style="display:none;"), selector = "#play2", where = "afterEnd")})
    observeEvent(input$play3, {insertUI(ui = tags$audio(src = "3.wav", type = " 'audio/wav", autoplay = NA, controls = NA, style="display:none;"), selector = "#play3", where = "afterEnd")})
    
    observeEvent(input$item1, {output$score1 <- renderText({paste0("You gave a score of ", input$item1)})})
    
    #observeEvent(input$submit, {output$ty <- renderText({paste0("Thanks for answering the survey ", input$name, "!")})})
    #observeEvent(input$scared, {insertUI(ui = img(src = "test.png", width = 350, height = 350), selector = "#scared", where = "afterEnd")})
    observeEvent(input$submit, {output$ty1 <- renderText({paste0("Thanks for answering the survey ", input$item1, input$item2, input$Click.Counter, "!")})})
    output$MainAction <- renderUI( {
        dynamicUi()
    })
    results = c(0)
    
    dynamicUi <- reactive({
        if (input$Click.Counter==0)
            return(
                list(
                    h3("Welcome to Deep Fakes Inc.!"),
                    h6("hej med dig")
                )
            )
        
        
        if (input$Click.Counter>0 & input$Click.Counter<=5 ){
            samples <<- sample(3, 3, replace = F)
            
            observeEvent(input$play11, {insertUI(ui = tags$audio(src = sprintf("%s.wav", samples[1]), type = " 'audio/wav", autoplay = NA, controls = NA, style="display:none;"), selector = "#play1", where = "afterEnd")})
            observeEvent(input$play21, {insertUI(ui = tags$audio(src = sprintf("%s.wav", samples[2]), type = " 'audio/wav", autoplay = NA, controls = NA, style="display:none;"), selector = "#play2", where = "afterEnd")})
            observeEvent(input$play31, {insertUI(ui = tags$audio(src = sprintf("%s.wav", samples[3]), autoplay = NA, controls = NA, style="display:none;"), selector = "#play3", where = "afterEnd")})
            return(
                list(
                    h3("Welcome to Deep Fakes Inc.!"),
                    h3(input$Click.Counter),
                    h3("Second sample"),
                    h4(samples[1], samples[2], samples[3]),
                    h6(results[length(results)]),
                    actionButton("play11", "Play sound 1"),
                    actionButton("play21", "Play sound 2"),
                    actionButton("play31", "Play sound 3"),
                    sliderInput("survey", "How well does it sound?", 0, 10, value = 5)
                    
                )
            )}
        if (input$Click.Counter>5)
            return(
                list(
                    h4("View aggregate results"),
                    h4("Thanks for taking the survey!")
                    
                )
            ) 
            
            
        
    })
    
}


shinyApp(ui = ui, server = server)
