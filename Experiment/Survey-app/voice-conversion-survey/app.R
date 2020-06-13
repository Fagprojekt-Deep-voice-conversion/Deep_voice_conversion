#load libraries
library(shiny)
library(shinyjs)
library(googlesheets4)
library(googledrive)
source("Combination.R")
options(gargle_oauth_cache = ".secrets")

drive_auth(cache = ".secrets", email = "peter@groenning.net")
gs4_auth(token = drive_token())

ui <- fluidPage(
    useShinyjs(),
    titlePanel("Deep Fakes inc."),
    sidebarPanel(
  
        h3("By us..."),
        h6(textOutput("save.results1")),
        h6(textOutput("save.results2")),
        h6(textOutput("checkcategory")),
        h6(textOutput("checkcategory1")),
        h6(textOutput("checkcategory2")),
        h6(textOutput("checkcategory3")),
        h6(textOutput("check")),
        numericInput("age", "Feel free to let us know your age", value = NaN, min = 0, max = 120),
        radioButtons("gender", "Feel free to let us know your gender", c("Male", "Female", "Other", "Prefer not to say"), selected = "Prefer not to say"),
        
        
      
      
       
       actionButton("play1", ""),
       actionButton("play2", ""),
       actionButton("play3", ""),
       actionButton("play4", "")
       
       ),
     
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

    observeEvent(input$converted1, {insertUI(ui = tags$audio(src = get_wavs_experiment(
                                                                       models[X[SamplesB,][input$Click.Counter - midways -2,][1]],
                                                                       categories[X[SamplesB,][input$Click.Counter - midways -2,][2]],
                                                                       subcategories[X[SamplesB,][input$Click.Counter - midways -2,][3]],
                                                                       "similarity", input$Click.Counter)[1],
                                                         type = " 'audio/wav", autoplay = NA,
                                                         controls = NA, style="display:none;"), selector = "#play1", where = "afterEnd")})
    
    observeEvent(input$target, {insertUI(ui = tags$audio(src = get_wavs_experiment(
                                                                       models[X[SamplesB,][input$Click.Counter- midways -2,][1]],
                                                                       categories[X[SamplesB,][input$Click.Counter- midways -2,][2]],
                                                                       subcategories[X[SamplesB,][input$Click.Counter- midways -2,][3]],
                                                                       "similarity", input$Click.Counter)[2],
                                                         type = " 'audio/wav", autoplay = NA,
                                                         controls = NA, style="display:none;"), selector = "#play1", where = "afterEnd")})

    observeEvent(input$converted2, {insertUI(ui = tags$audio(src = get_wavs_experiment(
                                                                       models[X[SamplesB,][input$Click.Counter- midways -2,][1]],
                                                                       categories[X[SamplesB,][input$Click.Counter- midways -2,][2]],
                                                                       subcategories[X[SamplesB,][input$Click.Counter- midways -2,][3]],
                                                                       "similarity", input$Click.Counter)[3],
                                                         type = " 'audio/wav", autoplay = NA,
                                                         controls = NA, style="display:none;"), selector = "#play1", where = "afterEnd")})
    
    
    observeEvent(input$real, {insertUI(ui = tags$audio(src = get_wavs_experiment(
                                                                       models[Y[SamplesA,][input$Click.Counter -1,][1]],
                                                                       categories[Y[SamplesA,][input$Click.Counter -1,][2]],
                                                                       subcategories[Y[SamplesA,][input$Click.Counter-1,][3]],
                                                                       "real_fake", input$Click.Counter)[1],
                                                         type = " 'audio/wav", autoplay = NA,
                                                         controls = NA, style="display:none;"), selector = "#play1", where = "afterEnd")})
    
    observeEvent(input$fake, {insertUI(ui = tags$audio(src = get_wavs_experiment(
                                                                     models[Y[SamplesA,][input$Click.Counter-1,][1]],
                                                                     categories[Y[SamplesA,][input$Click.Counter-1,][2]],
                                                                     subcategories[Y[SamplesA,][input$Click.Counter-1,][3]],
                                                                     "real_fake", input$Click.Counter)[2],
                                                       type = " 'audio/wav", autoplay = NA,
                                                       controls = NA, style="display:none;"), selector = "#play1", where = "afterEnd")})
    
    
    # observeEvent(input$item1, {output$score1 <- renderText({paste0("You gave a score of ", input$item1)})})
    

    
    
    
    output$MainAction <- renderUI( {
        dynamicUi()
    })
    
    
    n_questions <- 40
    midways <- 16
    
    models = c("AutoVC", "StarGAN", "Baseline")
    categories = c("Danish_Danish", "English_English", "20min", "10min", "Baseline")
    subcategories = c("Male_Male", "Female_Female", "Male_Female", "Female_Male", "Male_English", "Male_Danish", "Female_English", "Female_Danish")
  
    voices = c("source", "target", "converted")
    
    X <- combination()
    Y <- combination2()
    Y
    SamplesA
    SamplesA <- sample(nrow(Y), nrow(Y), replace = F)
    SamplesB <- sample(nrow(X), nrow(X), replace = F)
    # samples <- sample(3, 3, replace = F)
    
    
    real_fake <- sample(c("real", "fake"), 2, replace = F)
    
    similarityresults = rep(NaN, nrow(X))
    qualityresults = rep(NaN, nrow(X))
    fakenessresults = rep(NaN, nrow(Y))
    observe({toggleState("Click.Counter", condition = input$Check)})
    
    ss = "https://docs.google.com/spreadsheets/d/1Y2Hu04dY-chxSPdVgcUefXSTs6zvG6lkzAFlWhICPJA/edit#gid=0"
    
    dynamicUi <- reactive({
        if (input$Click.Counter==0){
        
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
                    # h3("| "),
                    # h3("| "),
                    
                    h4("We wish to investigate how far the deep fakes tehcnology is within voice conversion (faking peoples' voices)"),
                    h4("To do so, we wish to test two different methods for voice conversion on a set of different subtask with both english and danish voice"),
                    h4("This survey consists of 20 questions of 2 parts each. In the first part you are asked to rate the similarity of two voices and in the second part you are asked to rate the quality of a converted voice"),
                    h4("Please listen carefully to each audio file before rating."),
                    h4("Whenever you are ready click Next."),
                
                    checkboxInput("Check", value = FALSE, label = "Agree to terms and conditions"))
                )
          
        
            }
        
      if (input$Click.Counter == 1){
        return(list(
          h3("This is part 1")
        ))
      }
      if (input$Click.Counter>1 & input$Click.Counter<=midways+1 ){
        # real_fake <<- sample(c("real", "fake"), 2, replace = F)
        return(
          list(
            
            h2("Part 1: Which is real?"),
            
            h3("Question", input$Click.Counter-1),
            h4("A: ", real_fake[1], " B: ", real_fake[2]),
            h5("Guess which voice is real by pressing either A or B"),
            actionButton(real_fake[1], "Play sound A"),
            actionButton(real_fake[2], "Play sound B"),
            radioButtons("fakescore", "Which is real?", c("A", "B"))
            # actionButton("play31", "Play sound B"),
            # 
            # sliderInput("survey", "How similar is X to A or B?", min = -5, max = 5, value = 0),
            # 
            # h4("Part B: Quality"),
            # h5("Please rate the quality of this voice"),
            # h5("A score of 0 indicates the voice to not be understandable at all"),
            # h5("A score of 10 indicates the voice to be real i.e. not synthesized"),
            # actionButton("play41", "Play sound 4"),
            # sliderInput("survey1", "How well does it sound?", 0, 10, value = 5)
          )
        )}
        if (input$Click.Counter==midways+2 ){
          return(list(h2("Get ready for Part 2!")))
        }
        if (input$Click.Counter>midways + 2 & input$Click.Counter<=n_questions +2 ){
            source_target = sample(c("target","converted1"), 2, F)
            return(
                list(
                  h2("Part 2: Conversion Quality"),
                    h3("Question: ", input$Click.Counter -2),
                   
                    h4("Part A: Similarity"),
                    h5("Please rate the similarity of voice X with A and B"),
                     
                    h5("A score of 5 indicates with high confidence that A and B are the same voice"),
                    h5("A score of 0 indicates with high confidence that A and B are different voices"),
                    actionButton(source_target[1], "Play sound A"),
                    # actionButton("converted1", "Play sound X"),
                    actionButton(source_target[2], "Play sound B"),
                    
                    sliderInput("survey", "How similar are A and B?", min = 0, max = 5, value = 3),
                  
                    h4("Part B: Quality"),
                    h5("Please rate the naturalness of this voice"),
                    h5("A score of 0 indicates the voice to not be understandable at all"),
                    h5("A score of 5 indicates the voice to be real i.e. not synthesized"),
                    actionButton("converted2", "Play sound 4"),
                    sliderInput("survey1", "How well does it sound?", 0, 5, value = 3)
                )
            )}
        if (input$Click.Counter>n_questions + 2)
            return(
                list(
                    h3("Thanks for taking the survey!"),
                    
                    h4("This has truly been a great help to our project")
                    
                                    )
            ) 
            
    })
    output$checkcategory <- renderText({
        if (input$Click.Counter > 1 & input$Click.Counter <= midways + 1){
            return(
                models[Y[SamplesA,][input$Click.Counter-1,][1]])}
        else if (input$Click.Counter > 2 & input$Click.Counter <= n_questions +2){
            return(models[X[SamplesB,][input$Click.Counter -midways-2,][1]])}
        
        
        
    })
    output$checkcategory1 <- renderText({
        if (input$Click.Counter > 1 & input$Click.Counter <= midways + 1){
            return(
                categories[Y[SamplesA,][input$Click.Counter -1,][2]])}
        else if (input$Click.Counter > 2 & input$Click.Counter <= n_questions +2){
            return(categories[X[SamplesB,][input$Click.Counter - midways-2 ,][2]])}
        
        
        
    })
    output$checkcategory2 <- renderText({
      
        if (input$Click.Counter > 1 & input$Click.Counter <= midways + 1){
            return(
                subcategories[Y[SamplesA,][input$Click.Counter -1,][3]])}
        else if (input$Click.Counter > 2 & input$Click.Counter <= n_questions +2){
            return(subcategories[X[SamplesB,][input$Click.Counter - midways-2,][3]])}
        
        
        
    })

    
      
      
 
    
    output$Button <- renderText({
        if (input$Click.Counter >= n_questions + 2){
            return("Submit")
        }
        else{
            return("Next")
        }
    })
    score <- NaN
    output$save.results1 <- renderText({
      
      if ((input$Click.Counter>2) & (input$Click.Counter <= midways+2)){
        
        if (real_fake[1] == "fake"){
          if (input$fakescore == "A"){
            score <<- 0
          }
          else if (input$fakescore == "B"){
            score <<- 1
          }
          else{score <<- NaN}
          # try(fakenessresults[SamplesA[input$Click.Counter]]<<-score)
          
        }
        else if (real_fake[1] == "real"){
          if (input$fakescore == "A"){
            score <<- 1
          }
          else if (input$fakescore == "B"){
            score <<- 0
          }
          else{score <<- NaN}
          
          
        }
        try(fakenessresults[SamplesA[input$Click.Counter-2]]<<-score)
        # real_fake <<- sample(c("real", "fake"), 2, replace = F)
        # return(input$Click.Counter-2)
        real_fake <<- sample(c("real", "fake"), 2, replace = F)
      }
      
     return()
        
      
    })
    
    output$save.results2 <- renderText({
        # After each click, save the results of the radio buttons.
      
     
        if ((input$Click.Counter>midways+2)&(input$Click.Counter<=n_questions+2)){
            try(similarityresults[SamplesB[input$Click.Counter-midways-2]] <<- input$survey)
            try(qualityresults[SamplesB[input$Click.Counter-midways-2]] <<- input$survey1)
            return()}
      
        if (input$Click.Counter == n_questions + 3){
          
          try(sheet_append(ss, data.frame(t(c(input$age, input$gender, similarityresults))), sheet = "Similarity"))
          try(sheet_append(ss, data.frame(t(c(input$age, input$gender, similarityresults))), sheet = "Quality"))
          try(sheet_append(ss, data.frame(t(c(input$age, input$gender, fakenessresults))), sheet = "Fakeness"))
            return("Shiiit son you did it!")
        }
      
        
    })
    
  
    
    
  
   
    }
shinyApp(ui = ui, server = server)



