'''
Document Comparison (specifications from datasheets) using LangGraph

Created by: Bhanu
Modified on: 31st Dec 2024
'''

#........Importing Required Libraries........#
from io import BytesIO
import json
import pandas as pd
import fitz
import streamlit as st
from streamlit_chat import message
from typing_extensions import TypedDict
import openai
from openai import OpenAI
import numpy as np
from IPython.display import Image, display
import pprint

from langgraph.graph import END, START, StateGraph
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

#......Setting the environment variables manually......#
gpt_model = 'GPT_MODEL' #'gpt-3.5-turbo-0613' 
embedding_model = "EMBEDDING_MODEL"
client = OpenAI(api_key = "YOUR_KEYS")

# Function to extract text from PDF file
def extract_text_from_pdf(file) -> str:
    pdf_reader = fitz.open(stream=file.getvalue(), filetype="pdf")
    full_text = ""
    page_texts = []
    # for page_num in range(pdf_reader.getNumPages()):
    for page_num in range(pdf_reader.page_count):
        page = pdf_reader[page_num]
        text = page.get_text("text") 
        blocks = page.get_text("blocks")

        processed_blocks = []
        for b in blocks:
            block_text = b[4].strip()
            if block_text:
                if ":" in block_text:
                    processed_blocks.append(block_text)
                else:
                    processed_blocks.append(block_text.replace("\n", " "))
        processed_text = "\n".join(processed_blocks)
        full_text += processed_text + "\n\n"
        page_texts.append(processed_text)
    return page_texts


#......Function to extract specs from the input datasheet text......#
def extract_specs(pdf1_text: str, pdf2_text: str) -> str:
    text1 = pdf1_text
    text2 = pdf2_text

    # .....Generating the responses using the two input contexts......#
    response = client.chat.completions.create(
        model=gpt_model, 
        messages = [ {"role": "assistant", "content": """Identify the specifications from both the provided contexts and extract the values for the identified specifications from both the contexts. 
                      Compare the specifications from the provided two contexts and provide the comparison in a structured JSON format. Each specification should have an value entry for both contexts. 
                      The values should also include the associated units if available. If a specification is missing in one of the contexts, include 'N/A' for that entry.
                      The first two entires of the specification should be 'company' and 'product/Model Number'. Extract as many specifications as possible from both the provided contexts, even if they are not common to both.
                      Format your response as: 
                      {
                        "company": ["Value from context1", "Value from context2"],
                        "product/Model Number": ["Value from context1", "Value from context2"],
                        "Specification 3": ["Value from context1", "Value from context2"], 
                        "Specification 4": ["Value from context1", "Value from context2"], 
                        ... 
                      }
                      
                      """ 
                      },
                      {"role": "system", "content": f"context1: {text1}, context2: {text2}"},
                    #   {"role": "user", "content": query_text}
                      ], #prompt=prompt_template,
        max_tokens=1200,
        temperature = 0.1,
    )
    specifications = response.choices[0].message.content
    return specifications


###............Main Function.............###
def main():
    import time
    st.header("Specification Comparator with LangGraph ")
    
    # Upload PDF files 
    uploaded_file_1 = st.file_uploader("Upload First PDF", type="pdf")
    uploaded_file_2 = st.file_uploader("Upload Second PDF", type="pdf")
    
    submit=st.button("Submit")

    # Define the graph
    class State(TypedDict):
        uploaded_file_1: bytes
        uploaded_file_2: bytes
        pdf1_text: str
        pdf2_text: str
        specifications: str

    # Display uploaded files
    if submit and uploaded_file_1 and uploaded_file_2:
        with st.spinner("Comparing Documents.....Please Wait"):
            
            #..........Create the graph..........#
            graph_builder = StateGraph(State)

            #..........Define the nodes..........#
            graph_builder.add_node("extract_text_doc1", lambda state: {"pdf1_text": extract_text_from_pdf(state["uploaded_file_1"])})
            graph_builder.add_node("extract_text_doc2", lambda state: {"pdf2_text": extract_text_from_pdf(state["uploaded_file_2"])})
            graph_builder.add_node("extract_specs", lambda state: {"specifications": extract_specs(state["pdf1_text"], state["pdf2_text"])})
            
            # Define the edges that connects the nodes
            graph_builder.add_edge(START, "extract_text_doc1")
            graph_builder.add_edge("extract_text_doc1", "extract_text_doc2")
            graph_builder.add_edge("extract_text_doc2","extract_specs")
            graph_builder.add_edge("extract_specs", END)
            
            # Execute the graph
            graph = graph_builder.compile()
            # spec_list_1 = graph.run("extract_specs")

            # Assuming pdf1_file and pdf2_file are the uploaded PDF files in bytes
            input_state = {
                "uploaded_file_1": uploaded_file_1,
                "uploaded_file_2": uploaded_file_2
            }

            # Execute the graph
            result = graph.invoke(input_state)

            # Access the extracted specifications
            spec_list_1 = result["specifications"]

            # Parse the JSON response
            if spec_list_1:
                try:
                    data = json.loads(spec_list_1)
                    # st.write(data)
                    # Convert the JSON to a DataFrame
                    df = pd.DataFrame(data)
                    df2 = df.transpose().reset_index()
                    df2.columns = ['Specification', 'File 1', 'File 2']
                    print("JSON data loaded successfully.")

                    # Convert DataFrame to Excel
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df2.to_excel(writer, index=False, sheet_name='comparison_results')
                        writer.close()
                    excel_data = output.getvalue()
                    
                    # Provide download button
                    st.download_button(
                        label="Download data as Excel",
                        data=excel_data,
                        file_name='comparison_results.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

                    st.write(" ") 
                    st.write("Comparison Results as dataframe")
                    st.write(data)

                    #.....Code to display the outputs from each node of the graph......#
                    # for output in graph.stream(input_state):
                    #     for key, value in output.items():
                    #         pprint.pprint(f"Output from node '{key}':")
                    #         pprint.pprint("---")
                    #         pprint.pprint(value, indent=2, width=80, depth=None)
                    #     pprint.pprint("\n---\n")

                    #..........Displaying the generated Graph from LangGraph..........#
                    try:
                        # Visualize the graph
                        graph_image = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
                        # display(graph_image)
                        st.write("Graph generated by LangGraph")

                        # Create a download button for the image
                        # st.download_button(
                        #     label="Download Image",
                        #     data=graph_image,
                        #     file_name="graph.png",
                        #     mime="image/png"
                        # )
                        # display(graph_image)
                        # graph_image = graph_visualization.draw_mermaid_png(
                        #     draw_method=MermaidDrawMethod.API)
                    except Exception:
                        # st.error(f"An error occurred while generating the graph image: {e}")
                        graph_image = None
                        
                    if graph_image:
                        st.image(graph_image, caption="Graph generated by LangGraph", use_container_width=True)
                    else:
                        st.error("Graph image could not be generated.")

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
            else:
                print("spec_list_1 is empty or None.")
    else:
        st.write("Please upload the PDF files and click on Submit button to compare the documents")

if __name__=="__main__":
    main()