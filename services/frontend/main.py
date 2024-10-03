from components import App

if __name__ == "__main__":
    App().launch(
        share=False,
        ssl_verify=False,
        server_name="0.0.0.0",
        server_port=7860,
    )
