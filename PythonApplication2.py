import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os # Module to handle file paths and extensions

# --- 1. CORE DATA ANALYSIS FUNCTION ---

def load_data(file_path: str, timestamp_col: str, **kwargs) -> pd.DataFrame:
    """
    Dynamically loads data from CSV or Excel based on file extension.
    This function handles the crucial first step: getting data into a Pandas DataFrame.
    """
    if not file_path:
        # Stop and report an error if no file path was provided.
        raise FileNotFoundError("File path is empty.")
    
    # Get the file extension (e.g., '.csv' or '.xlsx') and convert to lowercase for reliable checking.
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.csv':
        # Load CSV (Comma Separated Values) file.
        # 'parse_dates' tells Pandas to treat the specified column as time data.
        df = pd.read_csv(file_path, parse_dates=[timestamp_col], **kwargs)
    elif ext == '.xlsx':
        # Load Excel file (requires the 'openpyxl' engine).
        df = pd.read_excel(file_path, parse_dates=[timestamp_col], **kwargs, engine='openpyxl')
    else:
        # Stop and report an error if the file type is not supported.
        raise ValueError(f"Unsupported file type: {ext}. Please use .csv or .xlsx.")
        
    return df

def run_rms_analysis(hires_file_path: str, weather_file_path: str) -> Dict[str, Any]:
    """
    The main analysis pipeline. It loads, cleans, merges, and analyzes 
    the bike hires and weather data to find trends.

    Returns:
        A dictionary containing key statistical results and aggregated data for charting.
    """
    
    # --- 1. Load Data ---
    try:
        # 1.1 Load hires data
        df_hires = load_data(
            hires_file_path,
            timestamp_col='timestamp',
            # Load 'hires' column as float initially to handle any non-numeric entries without crashing.
            dtype={'station_id': str, 'hires': float} 
        )
        # Convert 'hires' to a proper integer type (Int64) after loading, 
        # using 'coerce' to turn invalid values into NaN (missing data).
        df_hires['hires'] = pd.to_numeric(df_hires['hires'], errors='coerce').astype('Int64')

        # 1.2 Load weather data
        df_weather = load_data(
            weather_file_path,
            timestamp_col='timestamp'
        )
        
        # 1.3 Robustly convert 'temp_c' to numeric, handling non-numeric data gracefully.
        df_weather['temp_c'] = pd.to_numeric(df_weather['temp_c'], errors='coerce')

        # 1.4 Clean 'conditions' column (e.g., 'Rainy ', 'Sunny') by stripping whitespace.
        df_weather['conditions'] = df_weather['conditions'].astype(str).str.strip()

    except Exception as e:
        # If loading fails (e.g., file not found, wrong column name), raise a clear error.
        raise Exception(f"Error loading files. Check file format, column names, or installation of 'openpyxl' (for Excel). Details: {e}")


    # --- 2. Data Cleaning and Preprocessing ---
    
    # 2.1 Outlier Filtering (Temperature): Remove unrealistically cold or hot temperatures (e.g., -50°C or 50°C is usually a data error).
    initial_weather_rows = len(df_weather)
    df_weather = df_weather[
        (df_weather['temp_c'] >= -50) & 
        (df_weather['temp_c'] <= 50)
    ]
    cleaned_weather_rows = len(df_weather)
    
    # Check if any rows were dropped. (The original code includes a print statement here, useful for debugging).
    if initial_weather_rows != cleaned_weather_rows:
        print(f"WARNING: Dropped {initial_weather_rows - cleaned_weather_rows} rows from weather data due to unrealistic temperature values.")
        
    # 2.2 Handle Missing Values (NaN)
    # Remove any rows from hires data where 'hires' is missing (NaN).
    df_hires.dropna(subset=['hires'], inplace=True)
    # Remove any rows from weather data where 'temp_c' is missing (NaN).
    df_weather.dropna(subset=['temp_c'], inplace=True)

    # 2.3 Normalize Timestamps
    # 'dt.floor('H')' rounds the time down to the nearest hour (e.g., 10:15:30 becomes 10:00:00).
    # This is critical for merging, ensuring both files align perfectly on the hour.
    df_hires['timestamp'] = df_hires['timestamp'].dt.floor('H')
    df_weather['timestamp'] = df_weather['timestamp'].dt.floor('H')

    # 2.4 Drop potential duplicates that might skew the analysis.
    df_hires.drop_duplicates(subset=['timestamp', 'station_id'], inplace=True)
    df_weather.drop_duplicates(subset=['timestamp'], inplace=True)

    # --- 3. Merge DataFrames ---
    # Merge the hires and weather data based on the normalized 'timestamp' column.
    # 'how=inner' ensures we only keep records that exist in *both* files at the same hour.
    df_merged = pd.merge(df_hires, df_weather, on='timestamp', how='inner')
    
    if df_merged.empty:
        raise Exception("Merged DataFrame is empty. Ensure 'timestamp' columns match exactly (same hour/minute) in both files.")

    # --- 4. Feature Engineering ---
    # Create new, helpful columns from the timestamp for easy grouping later.
    df_merged['day_of_week'] = df_merged['timestamp'].dt.day_name() # e.g., 'Monday', 'Tuesday'
    df_merged['hour_of_day'] = df_merged['timestamp'].dt.hour       # e.g., 0 (midnight) to 23 (11 PM)
    df_merged['date'] = df_merged['timestamp'].dt.date             # Just the date part, used for daily totals


    # --- 5. Analysis and Aggregation ---

    # 5.1 Peak Day Analysis (Total Hires)
    # Calculate the total hires for each day.
    daily_demand = df_merged.groupby('date')['hires'].sum().reset_index()
    daily_demand['day_of_week'] = pd.to_datetime(daily_demand['date']).dt.day_name()
    
    # Define the correct order for days (Monday to Sunday) for presentation/charting.
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Summarize total hires by day of the week, ensuring all days are present (fillna(0)).
    peak_day_summary = daily_demand.groupby('day_of_week')['hires'].sum().reindex(day_order).fillna(0).reset_index()
    # Find the day with the maximum total hires.
    peak_day_result = peak_day_summary.loc[peak_day_summary['hires'].idxmax()]
    
    # 5.2 Peak Hour Analysis (Average Hires)
    # Calculate the average hires for each hour of the day across the entire dataset.
    peak_hour_summary = df_merged.groupby('hour_of_day')['hires'].mean().reset_index()
    # Find the hour with the maximum average hires.
    peak_hour_result = peak_hour_summary.loc[peak_hour_summary['hires'].idxmax()]

    # 5.3 Weather Correlation (Average Hires per Condition)
    # Calculate the average hires for each unique weather condition (e.g., 'Sunny', 'Drizzle').
    weather_demand = df_merged.groupby('conditions')['hires'].mean().reset_index().sort_values(by='hires', ascending=False)
    # The first row is the condition with the highest average hires.
    best_condition_result = weather_demand.iloc[0]

    # --- 6. Compile Results ---
    # Package all results and aggregated data into a single dictionary for the GUI.
    results = {
        'peak_day': peak_day_result['day_of_week'],
        'peak_day_total': int(peak_day_result['hires']),
        'peak_hour': int(peak_hour_result['hour_of_day']),
        'peak_hour_avg': peak_hour_result['hires'],
        'best_condition': best_condition_result['conditions'],
        'best_condition_avg': best_condition_result['hires'],
        'weather_demand': weather_demand.to_dict('records'),
        # DataFrames needed for Matplotlib charts:
        'chart_data_day': peak_day_summary,
        'chart_data_hour': peak_hour_summary
    }
    
    return results

# --- 2. CHARTING FUNCTION ---

def create_charts(results: Dict[str, Any], chart_frame: tk.Frame):
    """
    Generates and embeds Matplotlib charts (bar graphs) into the provided Tkinter frame.
    This creates the visual representation of the analysis results.
    """
    # Clear any previous charts from the frame before drawing new ones.
    for widget in chart_frame.winfo_children():
        widget.destroy()

    # --- Chart 1: Average Hires by Hour of Day ---
    # fig1 is the entire figure (the canvas), ax1 is the plot area (axes).
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    
    hour_data = results['chart_data_hour']
    ax1.bar(hour_data['hour_of_day'], hour_data['hires'], color='#007BFF')
    
    ax1.set_title('Average Hires by Hour of Day', fontsize=12)
    ax1.set_xlabel('Hour (0-23)')
    ax1.set_ylabel('Average Hires')
    ax1.set_xticks(hour_data['hour_of_day'][::2])  # Show labels every 2 hours
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Embed Chart 1 into the Tkinter frame using FigureCanvasTkAgg.
    canvas1 = FigureCanvasTkAgg(fig1, master=chart_frame)
    canvas_widget1 = canvas1.get_tk_widget()
    canvas_widget1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    # --- Chart 2: Total Hires by Day of Week ---
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    
    day_data = results['chart_data_day']
    ax2.bar(day_data['day_of_week'], day_data['hires'], color='#4CAF50')
    
    ax2.set_title('Total Hires by Day of Week', fontsize=12)
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Total Hires')
    ax2.tick_params(axis='x', rotation=45)  # Rotate labels to prevent overlap
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Embed Chart 2
    canvas2 = FigureCanvasTkAgg(fig2, master=chart_frame)
    canvas_widget2 = canvas2.get_tk_widget()
    canvas_widget2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    plt.tight_layout() # Adjust subplot parameters for a tight layout


# --- 3. GUI APPLICATION CLASS (Tkinter) ---

class RMSAnalysisApp:
    def __init__(self, master, default_hires_file, default_weather_file):
        """Initializes the main application window and all its widgets."""
        self.master = master
        master.title("RMS Data Analysis Tool")
        master.geometry("1000x800") 
        master.configure(bg='#f0f0f0')
        
        # Store default file paths.
        self.hires_file = default_hires_file
        self.weather_file = default_weather_file

        # Define styles for a cleaner look.
        font_main = ('Helvetica', 12)
        font_title = ('Helvetica', 18, 'bold')
        button_color = '#007BFF'
        button_active_color = '#0056b3'

        # --- Setup GUI Layout Frames ---
        
        # Header/Title frame
        header_frame = tk.Frame(master, bg='#f0f0f0')
        header_frame.pack(pady=10, fill='x', padx=20)
        tk.Label(header_frame, text="RMS Bike Hire & Weather Analysis", font=font_title, bg='#f0f0f0', fg='#333').pack()
        
        # Frame for file selection buttons and labels
        file_frame = tk.Frame(master, padx=10, pady=10, bg='#fff', relief=tk.RIDGE, bd=2)
        file_frame.pack(padx=20, pady=5, fill='x')

        # Define file types for the file dialog box.
        self.file_types = [
            ("Data files (CSV/Excel)", "*.csv *.xlsx"),
            ("CSV files", "*.csv"), 
            ("Excel files", "*.xlsx"), 
            ("All files", "*.*")
        ]

        # Hires File Selection Widgets (Labels, indicators, and buttons)
        tk.Label(file_frame, text="1. Bike Hires File (CSV/Excel):", font=font_main, bg='#fff').grid(row=0, column=0, sticky='w', padx=5)
        self.hires_label = tk.Label(file_frame, text=f"Selected: {self.hires_file}", fg='green', bg='#fff', anchor='w')
        self.hires_label.grid(row=1, column=0, sticky='ew', padx=5)
        tk.Button(file_frame, text="Browse Hires", command=self.select_hires_file, 
                  bg=button_color, fg='white', font=font_main, activebackground=button_active_color, relief=tk.RAISED).grid(row=1, column=1, padx=5)

        # Weather File Selection Widgets
        tk.Label(file_frame, text="2. Weather File (CSV/Excel):", font=font_main, bg='#fff').grid(row=2, column=0, sticky='w', pady=(10, 0), padx=5)
        self.weather_label = tk.Label(file_frame, text=f"Selected: {self.weather_file}", fg='green', bg='#fff', anchor='w')
        self.weather_label.grid(row=3, column=0, sticky='ew', padx=5)
        tk.Button(file_frame, text="Browse Weather", command=self.select_weather_file, 
                  bg=button_color, fg='white', font=font_main, activebackground=button_active_color, relief=tk.RAISED).grid(row=3, column=1, padx=5)

        file_frame.grid_columnconfigure(0, weight=1) # Makes the path label expand horizontally

        # Main Analysis Button
        self.analyze_button = tk.Button(master, text="3. Run Analysis", command=self.run_analysis_gui, 
                                        bg='#4CAF50', fg='white', font=('Helvetica', 14, 'bold'), 
                                        activebackground='#45a049', relief=tk.RAISED, state=tk.NORMAL)
        self.analyze_button.pack(padx=20, pady=10, fill='x')

        # Frame where Matplotlib charts will be embedded.
        tk.Label(master, text="Visual Analysis:", font=font_main, bg='#f0f0f0', fg='#333').pack(pady=(5, 0), padx=20, anchor='w')
        self.chart_frame = tk.Frame(master, bg='#fff', relief=tk.SUNKEN, bd=1)
        self.chart_frame.pack(padx=20, pady=5, fill='both', expand=True)

        # ScrolledText widget for the detailed report output.
        tk.Label(master, text="Detailed Report:", font=font_main, bg='#f0f0f0', fg='#333').pack(pady=(5, 0), padx=20, anchor='w')
        self.result_text = scrolledtext.ScrolledText(master, wrap=tk.WORD, height=10, font=('Consolas', 10), state=tk.DISABLED, bg='#fff', fg='#333')
        self.result_text.pack(padx=20, pady=(0, 20), fill='x')


    def select_hires_file(self):
        """Opens the standard file dialog for the user to select the hires data file."""
        filepath = filedialog.askopenfilename(
            defaultextension=".csv",
            filetypes=self.file_types
        )
        if filepath:
            self.hires_file = filepath
            filename = os.path.basename(filepath)
            self.hires_label.config(text=f"Selected: {filename}", fg='green')

    def select_weather_file(self):
        """Opens the standard file dialog for the user to select the weather data file."""
        filepath = filedialog.askopenfilename(
            defaultextension=".csv",
            filetypes=self.file_types
        )
        if filepath:
            self.weather_file = filepath
            filename = os.path.basename(filepath)
            self.weather_label.config(text=f"Selected: {filename}", fg='green')

    def run_analysis_gui(self):
        """
        The main GUI method called when the 'Run Analysis' button is pressed.
        It orchestrates the data analysis, charting, and report display.
        """
        # Reset the text area and indicate that processing has started.
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "--- Analysis Started (Loading and processing data)... ---\n\n")
        self.result_text.config(state=tk.DISABLED)
        
        try:
            # Disable the button and change text to prevent double-clicking.
            self.analyze_button.config(state=tk.DISABLED, text="Running Analysis...")

            # 1. Run Core Analysis (calls the data processing function)
            results = run_rms_analysis(self.hires_file, self.weather_file)
            
            # 2. Generate Charts (calls the charting function)
            create_charts(results, self.chart_frame)

            # 3. Format and Display Text Report
            report = self._format_results(results)

            self.result_text.config(state=tk.NORMAL)
            self.result_text.insert(tk.END, report)
            self.result_text.config(state=tk.DISABLED)
            
            # Re-enable the button and show success message.
            self.analyze_button.config(state=tk.NORMAL, text="3. Run Analysis")
            messagebox.showinfo("Success", "Analysis and Charts completed successfully!")

        except Exception as e:
            # Handle any error that occurred during loading or analysis.
            error_message = f"\n--- Analysis FAILED ---\n\nError: {e}\n\nPlease verify file integrity and column names (timestamp, hires, conditions, etc.). Note: Excel files may require the 'openpyxl' library."
            self.result_text.config(state=tk.NORMAL)
            self.result_text.insert(tk.END, error_message)
            self.result_text.config(state=tk.DISABLED)
            self.analyze_button.config(state=tk.NORMAL, text="3. Run Analysis (Error)")
            messagebox.showerror("Error", "Analysis failed. Check the report for details.")

    def _format_results(self, results: Dict[str, Any]) -> str:
        """Formats the analysis results (from the dictionary) into a readable string report."""
        
        report = ""
        report += "="*50 + "\n"
        report += "RMS DATA ANALYSIS REPORT\n"
        report += "="*50 + "\n\n"

        # 1. Peak Demand by Day
        report += "[1. Peak Demand by Day]\n"
        report += f"Highest Demand Day: **{results['peak_day']}** (Total Hires: {results['peak_day_total']:,})\n\n"
        
        # 2. Peak Demand by Hour
        report += "[2. Peak Demand by Hour]\n"
        report += f"Peak Demand Hour: **{results['peak_hour']:02d}:00** (Avg Hires: {results['peak_hour_avg']:.2f})\n\n" # :02d ensures 9 becomes 09
        
        # 3. Weather Correlation
        report += "[3. Weather Correlation]\n"
        report += f"Best Weather Condition for Hires: **{results['best_condition']}** (Avg Hires: {results['best_condition_avg']:.2f})\n"
        
        report += "\n--- Demand by Condition (Top 5) ---\n"
        # Display the top 5 weather conditions for easy reference.
        for item in results['weather_demand'][:5]:
            report += f"  - {item['conditions']:<20}: {item['hires']:.2f} Avg Hires\n" # :<20 formats the condition string to be left-aligned in 20 spaces

        report += "\n" + "="*50 + "\n"
        report += "Actionable Insight:\n"
        report += f"The visualization clearly shows a strong preference for bike hires during the early evening peak ({results['peak_hour']:02d}:00) and on {results['peak_day']}.\n"
        report += f"Resource allocation should focus heavily on these periods, especially when the forecast calls for **{results['best_condition']}** weather.\n"
        
        return report

# --- 4. Main Execution ---

if __name__ == "__main__":
    # Define default file names (these must exist in the directory or be selected by the user).
    HIRES_FILE = 'bike_hires_apr2025.xlsx - in.csv'
    WEATHER_FILE = 'weather_apr2025.xlsx - in.csv'

    # Tkinter setup: This starts the graphical user interface.
    root = tk.Tk()
    # Create an instance of the main application class.
    app = RMSAnalysisApp(root, HIRES_FILE, WEATHER_FILE)
    # Start the Tkinter event loop, which keeps the window open and interactive.
    root.mainloop()`1