library(shiny)
library(shinydashboard)
library(ggplot2)
library(magrittr)
library(dplyr)
source("data.R")


# UI
select_def <- c("DMM", "DFF", "DMF", "DFM", "EMM", "EFF", "EMF", "EFM")

header <- dashboardHeader(title = "VC results")

sidebar <- dashboardSidebar(
    sidebarMenu(
        menuItem("Similarity", tabName = "similarity", icon = icon("line-chart")),
        menuItem("Quality", tabName = "quality"),
        menuItem("Fool test", tabName = "fool"),
        menuItem("Overall", tabName = "overall")
    )
)


body <- dashboardBody(
    tabItems(
        tabItem(
            tabName = "similarity",
            h2("The similarity results"),
            fluidRow(
                box(plotOutput("plot_S", height = 250), downloadButton('dl_S_plot'), collapsible = T, collapsed = F),
                box(
                    box(checkboxGroupInput("conv_types_S", "", unique(S$conv_type), selected = select_def), collapsible = T, title = "Conversion types", collapsed = T),
                    box(checkboxGroupInput("gender_S", "", unique(S$Gender), selected = unique(S$Gender)), collapsible = T, title = "Gender", collapsed = T),
                    box(sliderInput("time_S", "", min = 0, max = 24, value = c(0, 24)), collapsible = T, title = "Time intervals", collapsed = T),
                    box(sliderInput("age_S", "", min = min(S$Age), max = max(S$Age), value = c(min(S$Age), max(S$Age))), collapsible = T, title = "Age interval", collapsed = T),
                    box(checkboxGroupInput("divide_S", "Should the data be divided in different groups?", c("True", "False"), selected = "True"), collapsible = T, title = "Divide data", collapsed = T),
                    collapsible = T, collapsed = T, title = "Specifications"
                )
               
            )
        ),
        tabItem(
            tabName = "quality",
            h2("The quality results"),
            fluidPage(
                box(plotOutput("plot_Q", height = 250), downloadButton('dl_Q_plot'), collapsible = T, collapsed = F),
                box(
                    box(checkboxGroupInput("conv_types_Q", "Conversion types", unique(Q$conv_type), selected = select_def), collapsible = T, title = "Conversion types", collapsed = T),
                    box(checkboxGroupInput("gender_Q", "", unique(Q$Gender), selected = unique(Q$Gender)), collapsible = T, title = "Gender", collapsed = T),
                    box(sliderInput("time_Q", "", min = 0, max = 24, value = c(0, 24)), collapsible = T, title = "Time intervals", collapsed = T),
                    collapsible = T, collapsed = T, title = "Specifications"
                )
                
            )
        ),
        tabItem(
            tabName = "fool",
            h2("The fool test results"),
            fluidPage(
                box(plotOutput("plot_F", height = 250), downloadButton('dl_F_plot'), collapsible = T, collapsed = F),
                box(
                    box(checkboxGroupInput("conv_types_F", "Conversion types", unique(Fool$conv_type), selected = select_def), collapsible = T, title = "Conversion types", collapsed = T),
                    box(checkboxGroupInput("gender_F", "", unique(Fool$Gender), selected = unique(Fool$Gender)), collapsible = T, title = "Gender", collapsed = T),
                    box(sliderInput("time_F", "", min = 0, max = 24, value = c(0, 24)), collapsible = T, title = "Time intervals", collapsed = T),
                    box(sliderInput("age_F", "", min = min(Fool$Age), max = max(Fool$Age), value = c(min(Fool$Age), max(Fool$Age))), collapsible = T, title = "Age interval", collapsed = T),
                    collapsible = T, collapsed = T, title = "Specifications"
                )
                
            )
        ),
        tabItem(
            tabName = "overall",
            h2("Overall stats"),
            fluidPage(
                textOutput("participants")
            )
        )
    )
)

ui <- dashboardPage(header, sidebar, body)


# Server
server <- function(input, output) {
    S_data <- reactive({
        t <- S %>% 
            filter(as.integer(hour(Time)) > input$time_S[1], as.integer(hour(Time)) < input$time_S[2]) %>% 
            filter(Gender %in% input$gender_S) %>% 
            filter(Age >= input$age_S[1], Age <= input$age_S[2]) %>% 
            group_by(model, conv_type, experiment) %>% 
            summarise(score = mean(score)) %>% 
            filter(conv_type %in% input$conv_types_S)
                
        t
    })
    
    Q_data <- reactive({
        t <- Q %>% 
            filter(as.integer(hour(Time)) > input$time_Q[1], as.integer(hour(Time)) < input$time_Q[2]) %>% 
            filter(Gender %in% input$gender_Q) %>% 
            filter(Age >= input$age_Q[1], Age <= input$age_Q[2]) %>%
            group_by(model, conv_type, experiment) %>% 
            summarise(score = mean(score)) %>% 
            filter(conv_type %in% input$conv_types_Q)
        t
    })
    
    F_data <- reactive({
        t <- Fool %>% 
            filter(as.integer(hour(Time)) > input$time_F[1], as.integer(hour(Time)) < input$time_F[2]) %>% 
            filter(Gender %in% input$gender_F) %>% 
            filter(Age >= input$age_F[1], Age <= input$age_F[2]) %>%
            group_by(model, conv_type, experiment) %>% 
            summarise(score = 1-mean(score)) %>% 
            filter(conv_type %in% input$conv_types_F)
        t
    })
    
    plot_S <- function(){
        ggplot(S_data(), aes(y = score, x = conv_type, fill = model)) +
            geom_bar(stat = "identity", position = "dodge") +
            ggtitle("Similarity") +
            theme(plot.title = element_text(size=20, face="bold", 
                                            margin = margin(10, 0, 10, 0))) +
            xlab("Conversion type") +
            ylab("Mean opinion score") +
            facet_grid(~experiment, scales = "free_x")
    }
    
    plot_Q <- function(){
        ggplot(Q_data(), aes(y = score, x = conv_type, fill = model)) +
            geom_bar(stat = "identity", position = "dodge") +
            ggtitle("Quality") +
            theme(plot.title = element_text(size=20, face="bold", 
                                            margin = margin(10, 0, 10, 0))) +
            xlab("Conversion type") +
            ylab("Mean opinion score")+
            facet_grid(~experiment, scales = "free_x")
    }
    
    plot_F <- function(){
        ggplot(F_data(), aes(y = score, x = conv_type, fill = model)) +
            geom_bar(stat = "identity", position = "dodge") +
            ggtitle("Fool test") +
            theme(plot.title = element_text(size=20, face="bold", 
                                            margin = margin(10, 0, 10, 0))) +
            xlab("Conversion type") +
            ylab("Percentage fooled")+
            facet_grid(~experiment, scales = "free_x")
    }
    
    
    
    output$plot_S <- renderPlot({plot_S()})
    output$plot_Q <- renderPlot({plot_Q()})
    output$plot_F <- renderPlot({plot_F()})
    
    output$dl_S_plot = downloadHandler(
        filename = 'similarity.png',
        content = function(file) {
            device <- function(..., width, height) {
                grDevices::png(..., width = width, height = height,
                               res = 300, units = "in")
            }
            ggsave(file, plot = plot_S(), device = device)
        })
    
    output$dl_Q_plot = downloadHandler(
        filename = 'quality.png',
        content = function(file) {
            device <- function(..., width, height) {
                grDevices::png(..., width = width, height = height,
                               res = 300, units = "in")
            }
            ggsave(file, plot = plot_Q(), device = device)
        })
    
    output$dl_F_plot = downloadHandler(
        filename = 'fooltest.png',
        content = function(file) {
            device <- function(..., width, height) {
                grDevices::png(..., width = width, height = height,
                               res = 300, units = "in")
            }
            ggsave(file, plot = plot_F(), device = device)
        })
    
    output$participants <- renderText({paste0("There have been ", n_participants, " participants so far")})
    
}


# Run the application 
shinyApp(ui = ui, server = server)
