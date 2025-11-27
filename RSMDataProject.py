import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os 


# --- THEME DEFINITIONS ---
# We define the colors here for easy switching.
LIGHT_THEME = {
    'bg_main': '#f0f0f0',     # Light grey background for the main window
    'bg_frame': '#ffffff',    # White background for contained frames/widgets
    'fg_text': '#333333',     # Dark text color
    'bg_input': '#e8e8e8',    # Light background for input/status indicators
    'btn_main': '#007BFF',    # Standard button color (Blue)
    'btn_run': '#4CAF50',     # Run button color (Green)
    'btn_text': 'white'
}

DARK_THEME = {
    'bg_main': '#2e2e2e',     # Dark grey background for the main window
    'bg_frame': '#3c3c3c',    # Slightly lighter dark background for contained frames
    'fg_text': '#ffffff',     # White text color
    'bg_input': '#4a4a4a',    # Darker background for input/status indicators
    'btn_main': '#6a9fdf',    # Standard button color (Light Blue)
    'btn_run': '#66bb6a',     # Run button color (Light Green)
    'btn_text': 'black'
}


# --- 1. CORE DATA ANALYSIS FUNCTION (NO CHANGES) ---

def load_data(file_path: str, timestamp_col: str, **kwargs) -> pd.DataFrame:
    """Dynamically loads data from CSV or Excel."""
    if not file_path:
        raise FileNotFoundError("File path is empty.")
    
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.csv':
        df = pd.read_csv(file_path, parse_dates=[timestamp_col], **kwargs)
    elif ext == '.xlsx':
        df = pd.read_excel(file_path, parse_dates=[timestamp_col], **kwargs, engine='openpyxl')
    else:
        raise ValueError(f"Unsupported file type: {ext}. Please use .csv or .xlsx.")
        
    return df

def run_rms_analysis(hires_file_path: str, weather_file_path: str) -> Dict[str, Any]:
    """The main analysis pipeline."""
    # ... (run_rms_analysis content remains the same) ...
    try:
        df_hires = load_data(
            hires_file_path,
            timestamp_col='timestamp',
            dtype={'station_id': str, 'hires': float} 
        )
        df_hires['hires'] = pd.to_numeric(df_hires['hires'], errors='coerce').astype('Int64')

        df_weather = load_data(
            weather_file_path,
            timestamp_col='timestamp'
        )
        
        df_weather['temp_c'] = pd.to_numeric(df_weather['temp_c'], errors='coerce')
        df_weather['conditions'] = df_weather['conditions'].astype(str).str.strip()

    except Exception as e:
        raise Exception(f"Error loading files. Check file format, column names, or installation of 'openpyxl' (for Excel). Details: {e}")


    # --- 2. Data Cleaning and Preprocessing ---
    initial_weather_rows = len(df_weather)
    df_weather = df_weather[
        (df_weather['temp_c'] >= -50) & 
        (df_weather['temp_c'] <= 50)
    ]
    cleaned_weather_rows = len(df_weather)
    
    if initial_weather_rows != cleaned_weather_rows:
        print(f"WARNING: Dropped {initial_weather_rows - cleaned_weather_rows} rows from weather data due to unrealistic temperature values.")
        
    df_hires.dropna(subset=['hires'], inplace=True)
    df_weather.dropna(subset=['temp_c'], inplace=True)

    df_hires['timestamp'] = df_hires['timestamp'].dt.floor('H')
    df_weather['timestamp'] = df_weather['timestamp'].dt.floor('H')

    df_hires.drop_duplicates(subset=['timestamp', 'station_id'], inplace=True)
    df_weather.drop_duplicates(subset=['timestamp'], inplace=True)

    # --- 3. Merge DataFrames ---
    df_merged = pd.merge(df_hires, df_weather, on='timestamp', how='inner')
    
    if df_merged.empty:
        raise Exception("Merged DataFrame is empty. Ensure 'timestamp' columns match exactly (same hour/minute) in both files.")

    # --- 4. Feature Engineering ---
    df_merged['day_of_week'] = df_merged['timestamp'].dt.day_name() 
    df_merged['hour_of_day'] = df_merged['timestamp'].dt.hour       
    df_merged['date'] = df_merged['timestamp'].dt.date             

    # --- 5. Analysis and Aggregation ---
    daily_demand = df_merged.groupby('date')['hires'].sum().reset_index()
    daily_demand['day_of_week'] = pd.to_datetime(daily_demand['date']).dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    peak_day_summary = daily_demand.groupby('day_of_week')['hires'].sum().reindex(day_order).fillna(0).reset_index()
    peak_day_result = peak_day_summary.loc[peak_day_summary['hires'].idxmax()]
    
    peak_hour_summary = df_merged.groupby('hour_of_day')['hires'].mean().reset_index()
    peak_hour_result = peak_hour_summary.loc[peak_hour_summary['hires'].idxmax()]

    weather_demand = df_merged.groupby('conditions')['hires'].mean().reset_index().sort_values(by='hires', ascending=False)
    best_condition_result = weather_demand.iloc[0]

    df_merged['temp_rounded'] = df_merged['temp_c'].round().astype(int)
    temp_demand = df_merged.groupby('temp_rounded')['hires'].mean().reset_index()
    peak_temp_result = temp_demand.loc[temp_demand['hires'].idxmax()]


    # --- 6. Compile Results ---
    results = {
        'peak_day': peak_day_result['day_of_week'],
        'peak_day_total': int(peak_day_result['hires']),
        'peak_hour': int(peak_hour_result['hour_of_day']),
        'peak_hour_avg': peak_hour_result['hires'],
        'best_condition': best_condition_result['conditions'],
        'best_condition_avg': best_condition_result['hires'],
        'weather_demand': weather_demand.to_dict('records'),
        'peak_temp': int(peak_temp_result['temp_rounded']),
        'peak_temp_avg': peak_temp_result['hires'],
        'chart_data_day': peak_day_summary,
        'chart_data_hour': peak_hour_summary,
        'chart_data_temp': temp_demand
    }
    
    return results

# --- 2. CHARTING FUNCTION ---

def create_charts(results: Dict[str, Any], chart_frame: tk.Frame, is_dark_mode: bool):
    """
    Generates and embeds Matplotlib charts into the provided Tkinter frame.
    It now accepts 'is_dark_mode' to style the charts appropriately.
    """
    for widget in chart_frame.winfo_children():
        widget.destroy()

    # Define chart colors based on the current mode
    plot_style = 'dark_background' if is_dark_mode else 'default'
    fig_color = DARK_THEME['bg_frame'] if is_dark_mode else LIGHT_THEME['bg_frame']
    text_color = DARK_THEME['fg_text'] if is_dark_mode else LIGHT_THEME['fg_text']
    
    # Use Matplotlib styles for better integration
    plt.style.use(plot_style)
    
    # Create a 1x3 grid of subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=fig_color) 

    # --- Chart 1: Day of Week ---
    ax1 = axes[0]
    day_data = results['chart_data_day']
    ax1.bar(day_data['day_of_week'], day_data['hires'], color='#4CAF50')
    ax1.set_title('Total Hires by Day of Week', fontsize=12, color=text_color)
    ax1.set_xlabel('Day of Week', color=text_color)
    ax1.set_ylabel('Total Hires', color=text_color)
    ax1.tick_params(axis='x', rotation=45, colors=text_color) 
    ax1.tick_params(axis='y', colors=text_color) 

    # --- Chart 2: Hour of Day ---
    ax2 = axes[1]
    hour_data = results['chart_data_hour']
    ax2.bar(hour_data['hour_of_day'], hour_data['hires'], color='#007BFF')
    ax2.set_title('Average Hires by Hour of Day', fontsize=12, color=text_color)
    ax2.set_xlabel('Hour (0-23)', color=text_color)
    ax2.set_ylabel('Average Hires', color=text_color)
    ax2.set_xticks(hour_data['hour_of_day'][::3]) 
    ax2.tick_params(axis='x', colors=text_color) 
    ax2.tick_params(axis='y', colors=text_color) 

    # --- Chart 3: Hires vs. Temperature ---
    ax3 = axes[2]
    temp_data = results['chart_data_temp']
    ax3.scatter(temp_data['temp_rounded'], temp_data['hires'], color='#FF5733', s=40, alpha=0.7)
    ax3.plot(temp_data['temp_rounded'], temp_data['hires'], color='r', linestyle='--', linewidth=1) 
    ax3.set_title('Avg Hires vs. Temperature (°C)', fontsize=12, color=text_color)
    ax3.set_xlabel('Temperature (°C)', color=text_color)
    ax3.set_ylabel('Average Hires', color=text_color)
    ax3.tick_params(axis='x', colors=text_color) 
    ax3.tick_params(axis='y', colors=text_color) 
    
    # Update axes facecolor and borders for better dark mode look
    for ax in axes:
        ax.set_facecolor(fig_color)
        plt.setp(ax.spines.values(), color=text_color)

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    plt.tight_layout() 


# --- 3. GUI APPLICATION CLASS (Tkinter) ---

class RMSAnalysisApp:
    def __init__(self, master, default_hires_file, default_weather_file):
        """Initializes the main application window and all its widgets, using tabs for output."""
        self.master = master
        master.title("RMS Data Analysis Tool")
        master.geometry("1200x850")
        
        self.hires_file = default_hires_file
        self.weather_file = default_weather_file
        self.current_theme = LIGHT_THEME  # Start in Light Mode
        self.is_dark_mode = False
        
        # Styles
        self.font_main = ('Helvetica', 12)
        self.font_title = ('Helvetica', 18, 'bold')
        self.file_types = [
            ("Data files (CSV/Excel)", "*.csv *.xlsx"),
            ("CSV files", "*.csv"), 
            ("Excel files", "*.xlsx"), 
            ("All files", "*.*")
        ]

        # --- Setup Header/Title frame and Toggle Button ---
        header_frame = tk.Frame(master)
        header_frame.pack(pady=10, fill='x', padx=20)
        
        tk.Label(header_frame, text="RMS Bike Hire & Weather Analysis (Tabbed Output)", font=self.font_title).pack(side=tk.LEFT)
        
        self.toggle_button = tk.Button(header_frame, text="🌙 Dark Mode", command=self.toggle_mode, 
                                        font=('Helvetica', 10), relief=tk.FLAT)
        self.toggle_button.pack(side=tk.RIGHT, padx=5)

        # --- Setup File Selection and Run Button (Top Section) ---
        self.file_frame = tk.Frame(master, relief=tk.RIDGE, bd=2)
        self.file_frame.pack(padx=20, pady=5, fill='x')

        # Hires File Widgets
        tk.Label(self.file_frame, text="1. Bike Hires File (CSV/Excel):", font=self.font_main).grid(row=0, column=0, sticky='w', padx=5)
        self.hires_label = tk.Label(self.file_frame, text=f"Selected: {self.hires_file}", fg='green', anchor='w')
        self.hires_label.grid(row=1, column=0, sticky='ew', padx=5)
        self.hires_btn = tk.Button(self.file_frame, text="Browse Hires", command=self.select_hires_file, font=self.font_main, fg='white')
        self.hires_btn.grid(row=1, column=1, padx=5)

        # Weather File Widgets
        tk.Label(self.file_frame, text="2. Weather File (CSV/Excel):", font=self.font_main).grid(row=2, column=0, sticky='w', pady=(10, 0), padx=5)
        self.weather_label = tk.Label(self.file_frame, text=f"Selected: {self.weather_file}", fg='green', anchor='w')
        self.weather_label.grid(row=3, column=0, sticky='ew', padx=5)
        self.weather_btn = tk.Button(self.file_frame, text="Browse Weather", command=self.select_weather_file, font=self.font_main, fg='white')
        self.weather_btn.grid(row=3, column=1, padx=5)

        self.file_frame.grid_columnconfigure(0, weight=1)

        self.analyze_button = tk.Button(master, text="3. Run Analysis", command=self.run_analysis_gui, 
                                        font=('Helvetica', 14, 'bold'), fg='white', relief=tk.RAISED, state=tk.NORMAL)
        self.analyze_button.pack(padx=20, pady=10, fill='x')

        # --- Setup Tabbed Output Area ---
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(padx=20, pady=(0, 20), fill='both', expand=True)

        # Chart Page
        self.chart_page = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_page, text='📈 Visual Analysis (Charts)')
        self.chart_frame = tk.Frame(self.chart_page, relief=tk.SUNKEN, bd=1)
        self.chart_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Report Page
        self.report_page = ttk.Frame(self.notebook)
        self.notebook.add(self.report_page, text='📝 Detailed Report (Text)')
        self.result_text = scrolledtext.ScrolledText(self.report_page, wrap=tk.WORD, height=10, font=('Consolas', 10), state=tk.DISABLED)
        self.result_text.pack(padx=10, pady=10, fill='both', expand=True)
        
        # Apply initial theme (Light Mode)
        self.apply_theme(self.current_theme)

    def apply_theme(self, theme: Dict[str, str]):
        """Applies the colors from the given theme dictionary to all widgets."""
        
        # 1. Root Window and Notebook/Tab Frames
        self.master.configure(bg=theme['bg_main'])
        
        # Since ttk widgets are styled differently, we only change the background of the contained tk.Frames
        self.file_frame.config(bg=theme['bg_frame'])
        self.chart_frame.config(bg=theme['bg_frame'])
        
        # Note: We don't change colors for self.chart_page and self.report_page because they are ttk.Frames 
        # which inherit theme style, and we only need to style their contents.

        # 2. Labels and Headings
        for widget in [self.master, self.file_frame]:
             for child in widget.winfo_children():
                if isinstance(child, tk.Label):
                    # We only change the bg/fg for standard tk.Label widgets, excluding the file path indicators
                    if child not in [self.hires_label, self.weather_label]:
                        child.config(bg=theme['bg_frame'], fg=theme['fg_text'])
        
        # Specifically configure the labels that show the file path
        self.hires_label.config(bg=theme['bg_frame'], fg=theme['fg_text'])
        self.weather_label.config(bg=theme['bg_frame'], fg=theme['fg_text'])
        
        # Configure the header label
        for child in self.master.winfo_children():
            if isinstance(child, tk.Frame) and child != self.file_frame:
                for sub_child in child.winfo_children():
                     if isinstance(sub_child, tk.Label):
                        sub_child.config(bg=theme['bg_main'], fg=theme['fg_text'])

        # 3. Buttons (Standard Tk Buttons)
        # Apply the main button style
        for btn in [self.hires_btn, self.weather_btn]:
            btn.config(bg=theme['btn_main'], fg=theme['btn_text'], activebackground=theme['bg_main'])
        
        # Apply the run button style
        self.analyze_button.config(bg=theme['btn_run'], fg=theme['btn_text'], activebackground=theme['bg_main'])
        
        # Toggle button style
        self.toggle_button.config(bg=theme['bg_main'], fg=theme['fg_text'], activebackground=theme['bg_main'])

        # 4. ScrolledText Report Area
        self.result_text.config(bg=theme['bg_input'], fg=theme['fg_text'], insertbackground=theme['fg_text'])

    def toggle_mode(self):
        """Switches between light and dark themes."""
        
        if self.is_dark_mode:
            # Switch to LIGHT mode
            self.current_theme = LIGHT_THEME
            self.is_dark_mode = False
            self.toggle_button.config(text="🌙 Dark Mode")
        else:
            # Switch to DARK mode
            self.current_theme = DARK_THEME
            self.is_dark_mode = True
            self.toggle_button.config(text="☀️ Light Mode")
            
        self.apply_theme(self.current_theme)
        
        # If analysis has already run, refresh charts to match the new theme
        if self.analyze_button['text'] == "3. Run Analysis":
             try:
                # Attempt to refresh the charts if the analysis results are available
                self.run_analysis_gui(update_charts_only=True)
             except Exception:
                # If there's an error (e.g., files not loaded), just ignore it
                pass
                
    def select_hires_file(self):
        """Opens the standard file dialog for the user to select the hires data file."""
        filepath = filedialog.askopenfilename(defaultextension=".csv", filetypes=self.file_types)
        if filepath:
            self.hires_file = filepath
            filename = os.path.basename(filepath)
            self.hires_label.config(text=f"Selected: {filename}")

    def select_weather_file(self):
        """Opens the standard file dialog for the user to select the weather data file."""
        filepath = filedialog.askopenfilename(defaultextension=".csv", filetypes=self.file_types)
        if filepath:
            self.weather_file = filepath
            filename = os.path.basename(filepath)
            self.weather_label.config(text=f"Selected: {filename}")

    def run_analysis_gui(self, update_charts_only=False):
        """
        The main GUI method called when the 'Run Analysis' button is pressed.
        It now includes a flag to only update the charts after a theme switch.
        """
        
        # If we are only updating charts, skip the expensive data loading and report generation steps
        if not update_charts_only:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "--- Analysis Started (Loading and processing data)... ---\n\n")
            self.result_text.config(state=tk.DISABLED)
            self.notebook.select(self.report_page) 
        
        try:
            if not update_charts_only:
                 self.analyze_button.config(state=tk.DISABLED, text="Running Analysis...")

            # 1. Run Core Analysis (or retrieve the previous results)
            # In a full app, you'd cache 'results'. For simplicity here, we re-run the analysis
            # only if we aren't in 'update_charts_only' mode (though re-running is costly).
            # A real-world solution would store the 'results' in self.
            results = run_rms_analysis(self.hires_file, self.weather_file)
            
            # 2. Generate Charts
            # Pass the current theme mode to style the chart elements
            create_charts(results, self.chart_frame, self.is_dark_mode)

            if not update_charts_only:
                # 3. Format and Display Text Report
                report = self._format_results(results)
                self.result_text.config(state=tk.NORMAL)
                self.result_text.insert(tk.END, report)
                self.result_text.config(state=tk.DISABLED)
                
                self.analyze_button.config(state=tk.NORMAL, text="3. Run Analysis")
                messagebox.showinfo("Success", "Analysis complete! Check the 'Visual Analysis' and 'Detailed Report' tabs.")
                self.notebook.select(self.chart_page) 

        except Exception as e:
            if not update_charts_only:
                error_message = f"\n--- Analysis FAILED ---\n\nError: {e}\n\nPlease verify file integrity and column names."
                self.result_text.config(state=tk.NORMAL)
                self.result_text.insert(tk.END, error_message)
                self.result_text.config(state=tk.DISABLED)
                self.analyze_button.config(state=tk.NORMAL, text="3. Run Analysis (Error)")
                messagebox.showerror("Error", "Analysis failed. Check the 'Detailed Report' tab for error specifics.")
            # Ignore errors if only updating charts, as the error was already shown

    def _format_results(self, results: Dict[str, Any]) -> str:
        """Formats the analysis results into a readable string report."""
        
        report = ""
        report += "="*50 + "\n"
        report += "RMS DATA ANALYSIS REPORT (ENHANCED)\n"
        report += "="*50 + "\n\n"
        report += "[1. Peak Demand by Day]\n"
        report += f"Highest Demand Day: **{results['peak_day']}** (Total Hires: {results['peak_day_total']:,})\n\n"
        report += "[2. Peak Demand by Hour]\n"
        report += f"Peak Demand Hour: **{results['peak_hour']:02d}:00** (Avg Hires: {results['peak_hour_avg']:.2f})\n\n" 
        report += "[3. Weather Condition Impact]\n"
        report += f"Best Weather Condition for Hires: **{results['best_condition']}** (Avg Hires: {results['best_condition_avg']:.2f})\n"
        report += "\n--- Demand by Condition (Top 5) ---\n"
        for item in results['weather_demand'][:5]:
            report += f"  - {item['conditions']:<20}: {item['hires']:.2f} Avg Hires\n" 
        report += "\n[4. Temperature Impact]\n"
        report += f"Optimal Temperature Range Peak: **{results['peak_temp']}°C** (Avg Hires: {results['peak_temp_avg']:.2f})\n\n"
        report += "\n" + "="*50 + "\n"
        report += "Actionable Insight: Resource Allocation Strategy\n"
        report += "="*50 + "\n"
        report += f"The analysis reveals demand peaks sharply during the **{results['peak_hour']:02d}:00** hour and on **{results['peak_day']}**.\n"
        report += f"The new weather analysis indicates a clear preference for riding in **{results['best_condition']}** weather and when the temperature is around **{results['peak_temp']}°C**.\n"
        report += "Resource allocation (bike redistribution and maintenance teams) should be maximized during the confluence of these factors to minimize stock-outs and maximize revenue.\n"
        
        return report

# --- 4. Main Execution ---

if __name__ == "__main__":
    HIRES_FILE = 'bike_hires_apr2025.xlsx - in.csv'
    WEATHER_FILE = 'weather_apr2025.xlsx - in.csv'

    root = tk.Tk()
    app = RMSAnalysisApp(root, HIRES_FILE, WEATHER_FILE)
    root.mainloop()