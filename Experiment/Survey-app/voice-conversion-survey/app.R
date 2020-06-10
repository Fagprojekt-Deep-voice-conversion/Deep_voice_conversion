#load libraries
library(shiny)
library(googlesheets4)
library(googledrive)
source("Combination.R")
options(gargle_oauth_cache = ".secrets")

drive_auth(cache = ".secrets", email = "peter@groenning.net")
gs4_auth(token = drive_token())

ui <- fluidPage(
    titlePanel("Deep Fakes inc."),
    sidebarPanel(
  
        h3("By Nazi Tubbies"),
        h6(textOutput("save.results")),
        h6(textOutput("checkcategory")),
        h6(textOutput("checkcategory1")),
        h6(textOutput("tester")),
        
        h6(textOutput("checkcategory2")),
    

       actionButton("play1", ""),
       actionButton("play2", ""),
       actionButton("play3", "")),
     
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

    observeEvent(input$play11, {insertUI(ui = tags$audio(src = sprintf("%s/%s/%s/%s.wav",
                                                                       models[X[Samples,][input$Click.Counter,][1]],
                                                                       categories[X[Samples,][input$Click.Counter,][2]],
                                                                       subcategories[X[Samples,][input$Click.Counter,][3]],
                                                                       samples[1]), type = " 'audio/wav", autoplay = NA,
                                                         controls = NA, style="display:none;"), selector = "#play1", where = "afterEnd")})
    
    observeEvent(input$play21, {insertUI(ui = tags$audio(src = sprintf("%s/%s/%s/%s.wav",
                                                                       models[X[Samples,][input$Click.Counter,][1]],
                                                                       categories[X[Samples,][input$Click.Counter,][2]],
                                                                       subcategories[X[Samples,][input$Click.Counter,][3]],
                                                                       samples[2]), type = " 'audio/wav", autoplay = NA,
                                                         controls = NA, style="display:none;"), selector = "#play1", where = "afterEnd")})

    observeEvent(input$play31, {insertUI(ui = tags$audio(src = sprintf("%s/%s/%s/%s.wav",
                                                                       models[X[Samples,][input$Click.Counter,][1]],
                                                                       categories[X[Samples,][input$Click.Counter,][2]],
                                                                       subcategories[X[Samples,][input$Click.Counter,][3]],
                                                                       samples[3]), type = " 'audio/wav", autoplay = NA,
                                                         controls = NA, style="display:none;"), selector = "#play1", where = "afterEnd")})
    
    
    observeEvent(input$play41, {insertUI(ui = tags$audio(src = sprintf("%s/%s/%s/%s.wav",
                                                                       models[X[Samples,][input$Click.Counter,][1]],
                                                                       categories[X[Samples,][input$Click.Counter,][2]],
                                                                       subcategories[X[Samples,][input$Click.Counter,][3]],
                                                                       samples[3]), type = " 'audio/wav", autoplay = NA,
                                                         controls = NA, style="display:none;"), selector = "#play1", where = "afterEnd")})
    
    
    
    observeEvent(input$item1, {output$score1 <- renderText({paste0("You gave a score of ", input$item1)})})
    
    
    
    output$MainAction <- renderUI( {
        dynamicUi()
    })

    X <- combination()

    n_questions <- 10
    models = c("AutoVC", "StarGAN", "Baseline")
    categories = c("Danish_Danish", "English_English", "Danish_English", "English_Danish", "5min", "2_5min", "0min", "Baseline")
    subcategories = c("Male_Male", "Female_Female", "Male_Female", "Female_Female", "Male_English", "Male_Danish", "Female_English", "Female_Danish")
    
    voices = c("source", "target", "converted")
    
    Samples <- sample(nrow(X), n_questions, replace = F)
    samples <- sample(4, 4, replace = F)
    
    
    
    similarityresults = rep(NaN, 64)
    qualityresults = rep(NaN, 64)
  
    
    ss = "https://docs.google.com/spreadsheets/d/1Y2Hu04dY-chxSPdVgcUefXSTs6zvG6lkzAFlWhICPJA/edit#gid=0"
    
    dynamicUi <- reactive({
        if (input$Click.Counter==0)
            return(
                list(
                    h2("Welcome to the Deep Fakes survey!"),
                    h4("Deep fakes are instances of fake news in which artificial intelligence is used
                       to synthesise realistic image and/or sound media."),
                    h4("A Deep fake could e.g. be an synthesised video with the image and sound of a famous person
                       or a person in power."),
                    h4("If this new technology is put to use by malicious actors it could have severe, negative consequences."),
                    h4("Imagine a synthesised video of a politician (or any other influential person) ridiculing himself/herself or making harmfull statements."),
                    h4("This video could shift the public opinion about this person dramatically or some people might act upon his/her fake statements."),
                    h3("| "),
                    h3("| "),
                    h4("We wish to investigate how far the deep fakes tehcnology is within voice conversion (faking peoples' voices)"),
                    h4("To do so, we wish to test two different methods for voice conversion on a set of different subtask with both english and danish voice"),
                    h4("This survey consists of 20 questions of 2 parts each. In the first part you are asked to rate the similarity of two voices and in the second part you are asked to rate the quality of a converted voice"),
                    h4("Please listen carefully to each audio file before rating."),
                    h4("Whenever you are ready click Next.")
                
                   
                )
            )
        
        
        if (input$Click.Counter>0 & input$Click.Counter<=n_questions ){
            
            return(
                list(

                    h3("Question: ", input$Click.Counter),
                   
                    h4("Part A: Similarity"),
                    h5("Please rate the similarity of voice X with A and B"),
                    h5("A score of -5 indicates with high confidence X is the same as A"), 
                    h5("A score of 5 indicates with high confidence X is the same as B"),
                    h5("A score of 0 indicates that X sounds like neither A nor B"),
                    actionButton("play11", "Play sound A"),
                    actionButton("play21", "Play sound X"),
                    actionButton("play31", "Play sound B"),
                    
                    sliderInput("survey", "How similar is X to A or B?", min = -5, max = 5, value = 0),
                  
                    h4("Part B: Quality"),
                    h5("Please rate the quality of this converted voice"),
                    h5("A score of 0 indicates the voice to not be understandable at all"),
                    h5("A score of 10 indicates the voice to be real i.e. not synthesized"),
                    actionButton("play41", "Play sound 4"),
                    sliderInput("survey1", "How well does it sound?", 0, 10, value = 5)
                )
            )}
        if (input$Click.Counter>n_questions)
            return(
                list(
                    h3("Thanks for taking the survey!"),
                    
                    h4("This has truly been a great help to our project")
                    # h4("Your results: "),
                    # h5("Similarity: ", similarityresults[1], similarityresults[2], 
                    #    similarityresults[3], similarityresults[4], similarityresults[5]),
                    # h5("Quality: ", qualityresults[1], qualityresults[2], 
                    #    qualityresults[3], qualityresults[4], qualityresults[5]),
                    # tableOutput("resultTable")
                )
            ) 
            
    })
    output$checkcategory <- renderText({
        if (input$Click.Counter <= n_questions){
            return(
                models[X[Samples,][input$Click.Counter,][1]])}
        else{
            return()}
        
        
        
    })
    output$checkcategory1 <- renderText({
        if (input$Click.Counter <= n_questions){
            return(
                categories[X[Samples,][input$Click.Counter,][2]])}
        else{
            return()}
        
        
        
    })
    output$checkcategory2 <- renderText({
        if (input$Click.Counter <= n_questions){
            return(
                subcategories[X[Samples,][input$Click.Counter,][3]])}
        else{
            return()}
        
        
        
    })
    
    output$Button <- renderText({
        if (input$Click.Counter >= n_questions){
            return("Submit")
        }
        else{
            return("Next")
        }
    })
    output$save.results <- renderText({
        # After each click, save the results of the radio buttons.
        if ((input$Click.Counter>0)&(input$Click.Counter<=n_questions)){
            try(similarityresults[Samples[input$Click.Counter]] <<- input$survey)
            try(qualityresults[Samples[input$Click.Counter]] <<- input$survey1)
            return()}
        if (input$Click.Counter == n_questions + 1){
          
          try(sheet_append(ss, data.frame(t(similarityresults)), sheet = "Similarity"))
          try(sheet_append(ss, data.frame(t(similarityresults)), sheet = "Quality"))
            return("Shiiit son you did it!")
        }
      
        
    })
    
  

    
    output$resultTable <- renderTable({
      return(cbind(similarityresults, qualityresults))
        })
   
    }
shinyApp(ui = ui, server = server)

