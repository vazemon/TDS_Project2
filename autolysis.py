# /// script
# requires.python = ">=3.13"
# dependencies = [
#    "requests",
#    "matplotlib",
#    "seaborn",
#    "pandas",
#    "chardet"
# ]
# ///


#importing required libraries
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
import json
import chardet
import base64


#setting global variables to bw used
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
MODEL = "gpt-4o-mini"
HEADERS = {
    "Content-type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }

llm_query_error_message = "Error querying llm"
original_dir = ""


#requests to llm
def run_llm_request(prompt, data, function, function_name):
    if function is None and function_name =="":
      json_data = {
          "model": MODEL,
          "messages": [{
              "role": "user",
              "content": [{
                  "type": "text",
                  "text": prompt
                  }
                  ] + [{
                  "type": "image_url",
                  "image_url": {
                      "detail": "low",
                      "url": f"data:image/png;base64,{image['content']}"
                      }
                  }
                  for image in data
                       ]
              }
          ]
      }
    else:
        json_data = {
            "model": MODEL,
            "response_format": {
                "type": "json_object"
                },
            "messages": [
                {"role":"system","content": prompt},
                {"role":"user", "content": data}
                ],
            "functions": json.loads(json.dumps(function)),
            "function_call": {
                "name": function_name
                }
        }

    try:
        response = requests.post(URL, headers = HEADERS, json = json_data)
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return llm_query_error_message
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return llm_query_error_message
    

#data loading
def load_data(filename):
    try:
        # Detect encoding
        with open(filename, mode='rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
        file_encoding = result['encoding']
        # Now open the file with the detected encoding and generate sample data
        with open(filename, mode = 'r', encoding = file_encoding) as f:
            sample_data = ''.join([f.readline() for i in range(15)])
        #load the whole dataset in a dataframe to work on
        df = pd.read_csv(filename, encoding = file_encoding)
        return df, sample_data
    except Exception as e:
        print(f"Error occured while reading the file: {e}")
        return None, None


#basic data analysis using llm
def analyse_data(data):
    #prompt for analysing data types
    prompt_analyse_data = (
        "Analyse the given dataset.The first line is the header and the subsequent lines are sample data."
        "Columns may have unclean data in them. Ignore those cells."
        "Infer the data type considering the values in each column and the column name."
        "Return a JSON object where each entry has column name and its inferred data type."
        "The response should directly use the get_column_type function." 
        "Supported column types are 'string', 'integer', 'float', 'datetime', 'boolean'.")
    #schema for ananlysing data
    function_get_column_type = [{
        "name": "get_column_type",
        "description": "Identify column names and their data types from given dataset.",
        "parameters":{ 
            "type": "object",
            "properties": {
                "Column_metadata": {
                    "type": "array",
                    "description": "Metadata for each column",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column_name": {
                                "type": "string",
                                "description": "Name of the column"
                                },
                            "column_type": {
                                "type": "string",
                                "description": "Data type of the column (eg. integer, string)",
                                "enum": ["string", "integer", "float", "datetime", "boolean"]
                            }
                        },
                        "required": ["column_name", "column_type"]
                    }
                }
            },
            "required" : ["Column_metadata"]
        }
    }]
    #llm call
    basic_data_analysis = run_llm_request(prompt_analyse_data, data, function_get_column_type, "get_column_type")
    if (basic_data_analysis == llm_query_error_message):
        print("llm call unsuccessful while data analysis")
        analysis_output = None
    else:
        analysis_output = json.loads(basic_data_analysis['choices'][0]['message']['function_call']['arguments'])['Column_metadata']
    
    return analysis_output


# function that call llm and generates charts out of the code sent by llm
def generate_charts(analysis_output, df, directory_name):
    #prompt for generating charts
    prompt_create_charts = (
        "Analyse given list of column names of a dataset."
        "Observe the column names and the data type mentioned and suggest multiple charts like bar chart, pie chart, line chart, histogram for these columns."
        "Make use of single column or multiple columns."
        "Give multiple suggestions of charts."
        "Return a JSON object where each entry has chart name and list of columns used for that chart, and a python code to create the chart."
        "The dataframe used in the python code should be referred as '''df'''"
        "The python code should not have any comments in it. Only use matplotlib and seaborn liabraries"
        "Create an appropriate chart and export the chart as png"
        "Use appropriate chart name considering columns getting used while generating png"
        "Return the same chart name that is used while generating png"
        "The code should not throw any warnings because of deprecated parameters."
        "The response should directly use the create_charts function.")
    #schema for chart generation
    function_create_charts = [{
        "name": "create_charts",
        "description": "Consider column names and their data types and suggest appropriate charts.",
        "parameters":{
            "type": "object",
            "properties": {
                "Charts": {
                    "type": "array",
                    "description": "Array of suggested charts",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chart_name": {
                                "type": "string",
                                "description": "Unique name of the chart (used for PNG filename)."
                                },
                            "columns_list": {
                                "type": "array",
                                "description": "List of columns to  be use for creating chart",
                                "items": {
                                    "type": "string",
                                    "description": "A single column name"
                                }
                            },
                            "chart_code": {
                                "type": "string",
                                "description": "Python code for suggested chart"
                            }
                        },
                        "required": ["chart_name", "columns_list","chart_code"]
                    },
                    "minItems": 1
                }
            },
            "required" : ["Charts"]
        }
    }]
    #call to llm
    create_chart_output = run_llm_request(prompt_create_charts, json.dumps(analysis_output), function_create_charts, "create_charts")
  
    if (create_chart_output == llm_query_error_message):
          print("llm call unsuccessful while create_chart call")
          return 0
    else:
        chart_output = json.loads(create_chart_output['choices'][0]['message']['function_call']['arguments'])['Charts']

    global original_dir
    original_dir = os.getcwd()
    os.makedirs(directory_name, exist_ok=True)
    os.chdir(directory_name)
    print("Current directory: ", os.getcwd())

    successfully_created_charts = 0
    if(len(chart_output) > 0):
        #creating charts and storing them as a csv file
        for output in range (len(chart_output)):
            try:
              print(chart_output[output]['chart_name'])
              print(chart_output[output]['columns_list'])
              chart_code = chart_output[output]['chart_code']
              exec(chart_code)
              print(f"Chart {output+1} successfully generated")
              successfully_created_charts += 1
              if(successfully_created_charts == 3):
                  break
            except:
                print(f"Error generating chart {output}")

    print("Total charts created", successfully_created_charts)
    os.chdir(original_dir)
    
    if(successfully_created_charts == 0):
      print("No charts generated")
      return 0
    else:
      print(f" {successfully_created_charts} charts generated successfully.")
      return successfully_created_charts


#function to generate md file
def generate_md(directory_name):
    #prompt md file generation
    prompt_markdown = (
        "Assume that you are a data scientist."
        "Data contains a few charts"
        "Analyse these charts and describe briefly about the following things:"
        "1. About the data"
        "2. Analysis of columns and their data types."
        "3. Insights from the statistics and the charts."
        "4. Where and how these insights can be applied"
        "write a narrative in Markdown format including insights and visualizations"
        )
    
    os.makedirs(directory_name, exist_ok=True)
    os.chdir(directory_name)
    current_dir = os.getcwd()
    print("Current directory: ", current_dir)

    created_charts = [file for file in os.listdir(current_dir) if file.endswith('.png')]
    #charts encoding
    encoded_images_data = []
    for png_file in created_charts:
        with open(png_file, 'rb') as file:
            encoded_image = base64.b64encode(file.read()).decode('utf-8')
            encoded_images_data.append({"filename": os.path.basename(png_file), "content": encoded_image})
    #sending encoded images to llm
    llm_md_response = run_llm_request(prompt_markdown, encoded_images_data, None, "")

    if (llm_md_response == llm_query_error_message):
        print("llm call unsuccessful while generating markdown")
        result = None
    else:
        readme_data = llm_md_response['choices'][0]['message']['content']
        with open(f"{current_dir}/README.md", "w") as file:
            file.write(readme_data)
        print("Markdown created successfully.")
        result = "Writing md successful."

    os.chdir(original_dir)
    return result
  

#main function
def main(csv_file_name):
    #directory name to store charts
    directory_name = csv_file_name.removesuffix(".csv")

    #data loading
    df, sample_data = load_data(csv_file_name)
    if df is None:
        print("Data loading failed")
        exit(1)

    #data analysis
    analysis_output= analyse_data(sample_data)
    if (analysis_output == None):
        print("Data analysis failed")
        exit(1)

    #send output of data analysis to generate charts for different columns
    charts_count = generate_charts(analysis_output, df, directory_name)

    attempts = 0
    while(charts_count == 0 and attempts < 5):
        charts_count = generate_charts(analysis_output, df, directory_name)
        attempts+= 1

    if(charts_count == 0 and attempts == 5):
        print("No charts generated")
        exit(1)

    if(charts_count > 0):
        output_generate_md = generate_md(directory_name)
        if(output_generate_md == None):
            print("Markdown generation was not performed")
            exit(1)
        else:
            print(output_generate_md)

#fetching file name through arguments
run_check = sys.argv
arg_length  = len(run_check)
if arg_length < 2:
    print("csv file name not specified")
    sys.exit(1)
elif arg_length > 2:
    print("too many arguments passed")
    sys.exit(1)
else:
    csv_file_name = run_check[1]


if __name__ == "__main__":
    main(csv_file_name)