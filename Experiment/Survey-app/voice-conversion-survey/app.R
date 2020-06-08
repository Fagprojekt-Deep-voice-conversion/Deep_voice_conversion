#load libraries
library(shiny)
# library(shinydashboard)


ui <- fluidPage(
    titlePanel("Deep fakes"),
    sidebarPanel(
        h3("Please enter your name"),
        textInput("name", ""),
        h6(textOutput("save.results")),
        h6(textOutput("checkcategory")),
        h6(textOutput("checkcategory1")),
    
       
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
        actionButton("submit1", "Submit"),
        textOutput("ty1")),
       
        #actionButton("scared", "Get haunted!")
        mainPanel(
            # Main Action is where most everything is happenning in the
            # object (where the welcome message, survey, and results appear)
            uiOutput("MainAction"),
            # This displays the action putton Next.
            actionButton("Click.Counter", textOutput("Button")), 
            
            textOutput("test")
        )
        
        
        
    
)

server <- function(input, output){
    # sliderValues <- reactive({
    #     data.frame(Name = c("Sound 1"),
    #                Value = c(input$item1),
    #                stringsAsFactors = F)})
    
    
    #observeEvent(input$Click.Counter, {output$ty <- renderText({paste0("You gave a score of ",counter)})})
    observeEvent(input$play11, {insertUI(ui = tags$audio(src = sprintf("%s/%s/%s.wav",
                                                                       categories[catsamples[input$Click.Counter]],
                                                                       subcategories[subcatsamples[input$Click.Counter]],
                                                                       samples[1]), type = " 'audio/wav", autoplay = NA,
                                                         controls = NA, style="display:none;"), selector = "#play1", where = "afterEnd")})
    
    observeEvent(input$play21, {insertUI(ui = tags$audio(src = sprintf("%s/%s/%s.wav", 
                                                                       categories[catsamples[input$Click.Counter]], 
                                                                       subcategories[subcatsamples[input$Click.Counter]], 
                                                                       samples[2]), type = " 'audio/wav", autoplay = NA, 
                                                         controls = NA, style="display:none;"), selector = "#play2", where = "afterEnd")})
    
    observeEvent(input$play31, {insertUI(ui = tags$audio(src = sprintf("%s/%s/%s.wav", 
                                                                       categories[catsamples[input$Click.Counter]],
                                                                       subcategories[subcatsamples[input$Click.Counter]],
                                                                       samples[3]), type = " 'audio/wav", autoplay = NA,
                                                         controls = NA, style="display:none;"), selector = "#play3", where = "afterEnd")})
    
    observeEvent(input$item1, {output$score1 <- renderText({paste0("You gave a score of ", input$item1)})})
    
    #observeEvent(input$submit, {output$ty <- renderText({paste0("Thanks for answering the survey ", input$name, "!")})})
    #observeEvent(input$scared, {insertUI(ui = img(src = "test.png", width = 350, height = 350), selector = "#scared", where = "afterEnd")})
    observeEvent(input$submit1, {output$ty1 <- renderText({paste0("Thanks for answering the survey ", input$item1, input$item2, input$Click.Counter, "!")})})
    observeEvent(input$submit, {output$test <- renderText({paste0("Your number of clicks: ", input$Click.Counter, results, input$survey, "! omg crazy")})})
    output$MainAction <- renderUI( {
        dynamicUi()
    })
    
    sliderresults = rep(0, 5)
    catresults = rep("", 5)
    
    categories = c("Danish_Danish", "English_English", "Danish_English", "English_Danish")
    subcategories = c("Male_Male", "Female_Female", "Male_Female", "Female_Female")
    
    samples <<- sample(3, 3, replace = F)
    catsamples <<- sample(4,5, replace = T)
    subcatsamples <<- sample(4,5, replace = T)
    
    dynamicUi <- reactive({
        if (input$Click.Counter==0)
            return(
                list(
                    h3("Welcome to Deep Fakes Inc.!"),
                    h6("hej med dig")
                   
                )
            )
        
        
        if (input$Click.Counter>0 & input$Click.Counter<=5 ){
            
            
            #observeEvent(eval(parse(text=sprintf("input$play1%s", input$Click.Counter))), {insertUI(ui = tags$audio(src = sprintf("%s.wav", samples[1]), type = " 'audio/wav", autoplay = NA, controls = NA, style="display:none;"), selector = "#play1", where = "afterEnd")})
            #observeEvent(eval(parse(text=sprintf("input$play2%s", input$Click.Counter))), {insertUI(ui = tags$audio(src = sprintf("%s.wav", samples[2]), type = " 'audio/wav", autoplay = NA, controls = NA, style="display:none;"), selector = "#play1", where = "afterEnd")})
            #observeEvent(eval(parse(text=sprintf("input$play3%s", input$Click.Counter))), {insertUI(ui = tags$audio(src = sprintf("%s.wav", samples[3]), type = " 'audio/wav", autoplay = NA, controls = NA, style="display:none;"), selector = "#play1", where = "afterEnd")})
            
            #observeEvent(input$play21, {insertUI(ui = tags$audio(src = sprintf("%s.wav", samples[2]), type = " 'audio/wav", autoplay = NA, controls = NA, style="display:none;"), selector = "#play2", where = "afterEnd")})
            #observeEvent(input$play31, {insertUI(ui = tags$audio(src = sprintf("%s.wav", samples[3]), autoplay = NA, controls = NA, style="display:none;"), selector = "#play3", where = "afterEnd")})
            return(
                list(
                    h3("Welcome to Deep Fakes Inc.!"),
                    h3(input$Click.Counter),
                    h3("Sound test number: ", input$Click.Counter),
                    h4(samples[1], samples[2], samples[3]),
                    h6(results[length(results)]),
                    actionButton("play11", "Play sound 1"),
                    actionButton("play21", "Play sound 2"),
                    actionButton("play31", "Play sound 3"),
                    sliderInput("survey", "How well does it sound?", 0, 10, value = 5),
                    actionButton(sprintf("Click%s", input$Click.Counter), "Click Me")
                    
                )
            )}
        if (input$Click.Counter>5)
            return(
                list(
                    h3("Welcome to Deep Fakes Inc.!"),
                    h3(input$Click.Counter),
                    h3("Sound test number: ", input$Click.Counter),
                    h4(samples[1], samples[2], samples[3]),
                    h6(results[length(results)]),
                    actionButton("play11", "Play sound 1"),
                    actionButton("play21", "Play sound 2"),
                    actionButton("play31", "Play sound 3"),
                    h4("Thanks for taking the survey!"),
                    h5("Your results: ", sliderresults[1], sliderresults[2], 
                       sliderresults[3], sliderresults[4], sliderresults[5])
                )
            ) 
            
    })
    output$checkcategory <- renderText({
        if (input$Click.Counter <= 5){
        return(
                categories[catsamples[input$Click.Counter]])}
        else{
            return()}
        
        
            
    })
    output$checkcategory1 <- renderText({
        if (input$Click.Counter <= 5){
        return(
            subcategories[subcatsamples[input$Click.Counter]])}
        else{
            return()}
        
        
        
    })
    output$Button <- renderText({
        if (input$Click.Counter >= 5){
            return("Submit")
        }
        else{
            return("Next")
        }
    })
    output$save.results <- renderText({
        # After each click, save the results of the radio buttons.
        if ((input$Click.Counter>0)&(input$Click.Counter<=5)){
            try(sliderresults[input$Click.Counter] <<- input$survey)
            return()}
    })
    output$save.results1 <- renderText({
        # After each click, save the results of the radio buttons.
        if ((input$Click.Counter>0)&(input$Click.Counter<=5)){
            try(sliderresults[input$Click.Counter] <<- input$survey)
            return()}
        
    })
    A = 0
    observeEvent(eval(parse(text=sprintf("input$Click%s", input$Click.Counter))), {A = A + 1}) 
    output$A <- renderText({return(A)})
}


shinyApp(ui = ui, server = server)

