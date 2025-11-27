import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os 


# --- THEME DEFINITIONS (NO CHANGE) ---
LIGHT_THEME = {
    'bg_main': '#f0f0f0', 'bg_frame': '#ffffff', 'fg_text': '#333333', 
    'bg_input': '#e8e8e8', 'btn_main': '#007BFF', 'btn_run': '#4CAF50', 
    'btn_text': 'white'
}

DARK_THEME = {
    'bg_main': '#2e2e2e', 'bg_frame': '#3c3c3c', 'fg_text': '#ffffff', 
    'bg_input': '#4a4a4a', 'btn_main': '#6a9fdf', 'btn_run': '#66bb6a', 
    'btn_text': 'black'
}


# --- 1. CORE DATA ANALYSIS FUNCTION (REVISED FOR CSV ONLY) ---

def load_data(file_path: str, timestamp_col: str, **kwargs) -> pd.DataFrame:
    """Loads data from CSV only."""
    if not file_path:
        raise FileNotFoundError("File path is empty.")
    
    ext = os.path.splitext(file_path)[1].lower()

    # --- CHANGE: Only CSV is supported ---
    if ext == '.csv':
        # Use low_memory=False to prevent mixed type warnings on large files
        df = pd.read_csv(file_path, parse_dates=[timestamp_col], **kwargs, low_memory=False)
    else:
        # This should ideally not be reached since the file dialog is restricted
        raise ValueError(f"Unsupported file type: {ext}. Please use .csv.")
        
    return df

def run_rms_analysis(hires_file_path: str, weather_file_path: str) -> Dict[str, Any]:
    """The main analysis pipeline."""
    try:
        # Load and clean hires data
        df_hires = load_data(
            hires_file_path,
            timestamp_col='timestamp',
            dtype={'station_id': str, 'hires': float} 
        )
        df_hires['hires'] = pd.to_numeric(df_hires['hires'], errors='coerce').astype('Int64')

        # Load and clean weather data
        df_weather = load_data(
            weather_file_path,
            timestamp_col='timestamp'
        )
        # ... (rest of the run_rms_analysis logic remains the same) ...
        df_weather['temp_c'] = pd.to_numeric(df_weather['temp_c'], errors='coerce')
        df_weather['conditions'] = df_weather['conditions'].astype(str).str.strip()

    except Exception as e:
        # Updated error message to reflect CSV-only format
        raise Exception(f"Error loading files. Check file format (must be CSV), column names, or data types. Details: {e}")


    # --- Data Cleaning, Merge, Feature Engineering, and Analysis ---
    df_weather = df_weather[(df_weather['temp_c'] >= -50) & (df_weather['temp_c'] <= 50)]
    df_hires.dropna(subset=['hires'], inplace=True)
    df_weather.dropna(subset=['temp_c'], inplace=True)
    df_hires['timestamp'] = df_hires['timestamp'].dt.floor('H')
    df_weather['timestamp'] = df_weather['timestamp'].dt.floor('H')
    df_merged = pd.merge(df_hires, df_weather, on='timestamp', how='inner')
    
    if df_merged.empty:
        raise Exception("Merged DataFrame is empty. Ensure 'timestamp' columns match exactly (same hour/minute) in both files.")

    df_merged['day_of_week'] = df_merged['timestamp'].dt.day_name() 
    df_merged['hour_of_day'] = df_merged['timestamp'].dt.hour       
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

    results = {
        'peak_day': peak_day_result['day_of_week'], 'peak_day_total': int(peak_day_result['hires']),
        'peak_hour': int(peak_hour_result['hour_of_day']), 'peak_hour_avg': peak_hour_result['hires'],
        'best_condition': best_condition_result['conditions'], 'best_condition_avg': best_condition_result['hires'],
        'weather_demand': weather_demand.to_dict('records'),
        'peak_temp': int(peak_temp_result['temp_rounded']), 'peak_temp_avg': peak_temp_result['hires'],
        'chart_data_day': peak_day_summary, 'chart_data_hour': peak_hour_summary, 'chart_data_temp': temp_demand
    }
    return results

# --- 2. CHARTING FUNCTION (NO CHANGE) ---

def create_charts(results: Dict[str, Any], chart_frame: tk.Frame, is_dark_mode: bool):
    # ... (Charting logic remains the same) ...
    for widget in chart_frame.winfo_children():
        widget.destroy()

    plot_style = 'dark_background' if is_dark_mode else 'default'
    fig_color = DARK_THEME['bg_frame'] if is_dark_mode else LIGHT_THEME['bg_frame']
    text_color = DARK_THEME['fg_text'] if is_dark_mode else LIGHT_THEME['fg_text']
    
    plt.style.use(plot_style)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=fig_color) 

    # Chart 1: Day of Week
    ax1 = axes[0]
    day_data = results['chart_data_day']
    ax1.bar(day_data['day_of_week'], day_data['hires'], color='#4CAF50')
    ax1.set_title('Total Hires by Day of Week', fontsize=12, color=text_color)
    ax1.set_xlabel('Day of Week', color=text_color)
    ax1.set_ylabel('Total Hires', color=text_color)
    ax1.tick_params(axis='x', rotation=45, colors=text_color) 
    ax1.tick_params(axis='y', colors=text_color) 

    # Chart 2: Hour of Day
    ax2 = axes[1]
    hour_data = results['chart_data_hour']
    ax2.bar(hour_data['hour_of_day'], hour_data['hires'], color='#007BFF')
    ax2.set_title('Average Hires by Hour of Day', fontsize=12, color=text_color)
    ax2.set_xlabel('Hour (0-23)', color=text_color)
    ax2.set_ylabel('Average Hires', color=text_color)
    ax2.set_xticks(hour_data['hour_of_day'][::3]) 
    ax2.tick_params(axis='x', colors=text_color) 
    ax2.tick_params(axis='y', colors=text_color) 

    # Chart 3: Hires vs. Temperature
    ax3 = axes[2]
    temp_data = results['chart_data_temp']
    ax3.scatter(temp_data['temp_rounded'], temp_data['hires'], color='#FF5733', s=40, alpha=0.7)
    ax3.plot(temp_data['temp_rounded'], temp_data['hires'], color='r', linestyle='--', linewidth=1) 
    ax3.set_title('Avg Hires vs. Temperature (°C)', fontsize=12, color=text_color)
    ax3.set_xlabel('Temperature (°C)', color=text_color)
    ax3.set_ylabel('Average Hires', color=text_color)
    ax3.tick_params(axis='x', colors=text_color) 
    ax3.tick_params(axis='y', colors=text_color) 
    
    for ax in axes:
        ax.set_facecolor(fig_color)
        plt.setp(ax.spines.values(), color=text_color)

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    plt.tight_layout() 


# --- 3. GUI APPLICATION CLASS (Tkinter) (REVISED) ---

class RMSAnalysisApp:
    def __init__(self, master, default_hires_file, default_weather_file):
        """Initializes the main application window and all its widgets."""
        self.master = master
        master.title("RMS Data Analysis Tool")
        master.geometry("1200x850")
        
        # --- CHANGE 1: Default files are empty strings, so the button starts disabled ---
        self.hires_file = ""
        self.weather_file = ""
        
        self.current_theme = LIGHT_THEME  
        self.is_dark_mode = False        
        
        self.font_main = ('Helvetica', 12)
        self.font_title = ('Helvetica', 18, 'bold')
        
        # --- CHANGE 2: Restrict file types to CSV only ---
        self.file_types = [
            ("CSV files", "*.csv"), 
            ("All files", "*.*")
        ]
        
        self.style = ttk.Style()
        self.configure_ttk_style_light()

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
        tk.Label(self.file_frame, text="1. Bike Hires File (CSV Only):", font=self.font_main).grid(row=0, column=0, sticky='w', padx=5)
        # Displaying a helpful prompt until a file is selected
        self.hires_label = tk.Label(self.file_frame, text="Selected: None", fg='red', anchor='w')
        self.hires_label.grid(row=1, column=0, sticky='ew', padx=5)
        self.hires_btn = tk.Button(self.file_frame, text="Browse Hires", command=self.select_hires_file, font=self.font_main, fg='white')
        self.hires_btn.grid(row=1, column=1, padx=5)

        # Weather File Widgets
        tk.Label(self.file_frame, text="2. Weather File (CSV Only):", font=self.font_main).grid(row=2, column=0, sticky='w', pady=(10, 0), padx=5)
        self.weather_label = tk.Label(self.file_frame, text="Selected: None", fg='red', anchor='w')
        self.weather_label.grid(row=3, column=0, sticky='ew', padx=5)
        self.weather_btn = tk.Button(self.file_frame, text="Browse Weather", command=self.select_weather_file, font=self.font_main, fg='white')
        self.weather_btn.grid(row=3, column=1, padx=5)

        self.file_frame.grid_columnconfigure(0, weight=1)

        # --- CHANGE 3: Run Button starts disabled ---
        self.analyze_button = tk.Button(master, text="3. Run Analysis", command=self.run_analysis_gui, 
                                        font=('Helvetica', 14, 'bold'), fg='white', relief=tk.RAISED, state=tk.DISABLED)
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
        
        self.apply_theme(self.current_theme)

    def check_ready_to_run(self):
        """Checks if both files have valid paths and enables/disables the Run button."""
        if self.hires_file and self.weather_file:
            self.analyze_button.config(state=tk.NORMAL)
        else:
            self.analyze_button.config(state=tk.DISABLED)

    def select_hires_file(self):
        """Opens the file dialog and updates the button state."""
        filepath = filedialog.askopenfilename(defaultextension=".csv", filetypes=self.file_types)
        if filepath:
            self.hires_file = filepath
            filename = os.path.basename(filepath)
            self.hires_label.config(text=f"Selected: {filename}", fg='green')
        else:
            # If the user cancels the dialog, ensure the state is checked
            self.hires_label.config(text="Selected: None", fg='red')
            
        self.check_ready_to_run()

    def select_weather_file(self):
        """Opens the file dialog and updates the button state."""
        filepath = filedialog.askopenfilename(defaultextension=".csv", filetypes=self.file_types)
        if filepath:
            self.weather_file = filepath
            filename = os.path.basename(filepath)
            self.weather_label.config(text=f"Selected: {filename}", fg='green')
        else:
            self.weather_label.config(text="Selected: None", fg='red')
            
        self.check_ready_to_run()


    # --- Remaining helper methods (toggle_mode, apply_theme, _format_results) ---

    def configure_ttk_style_dark(self):
        self.style.configure('TNotebook', background=DARK_THEME['bg_main'], borderwidth=0, bordercolor=DARK_THEME['bg_main'], tabmargins=[2, 5, 2, 0])
        self.style.configure('TNotebook.Tab', background=DARK_THEME['bg_frame'], foreground=DARK_THEME['fg_text'], padding=[10, 5])
        self.style.map('TNotebook.Tab', background=[('selected', DARK_THEME['bg_main'])], foreground=[('selected', DARK_THEME['fg_text'])], expand=[('selected', [1, 1, 1, 0])])
        self.style.configure('TFrame', background=DARK_THEME['bg_main'])


    def configure_ttk_style_light(self):
        self.style.configure('TNotebook', background=LIGHT_THEME['bg_main'], borderwidth=0, bordercolor=LIGHT_THEME['bg_main'], tabmargins=[2, 5, 2, 0])
        self.style.configure('TNotebook.Tab', background=LIGHT_THEME['bg_frame'], foreground=LIGHT_THEME['fg_text'], padding=[10, 5])
        self.style.map('TNotebook.Tab', background=[('selected', LIGHT_THEME['bg_main'])], foreground=[('selected', LIGHT_THEME['fg_text'])], expand=[('selected', [1, 1, 1, 0])])
        self.style.configure('TFrame', background=LIGHT_THEME['bg_main'])


    def apply_theme(self, theme: Dict[str, str]):
        if self.is_dark_mode:
            self.configure_ttk_style_dark()
        else:
            self.configure_ttk_style_light()
        
        self.master.configure(bg=theme['bg_main'])
        self.file_frame.config(bg=theme['bg_frame'])
        self.chart_frame.config(bg=theme['bg_frame'])

        label_widgets = []
        for parent in [self.master, self.file_frame]:
             for child in parent.winfo_children():
                if isinstance(child, tk.Label):
                    label_widgets.append(child)
                if isinstance(child, tk.Frame):
                    for sub_child in child.winfo_children():
                        if isinstance(sub_child, tk.Label):
                             label_widgets.append(sub_child)

        for label in label_widgets:
            bg_color = theme['bg_frame'] if label.master == self.file_frame else theme['bg_main']
            # Preserve the red/green status color for the path labels
            if label not in [self.hires_label, self.weather_label]:
                 label.config(bg=bg_color, fg=theme['fg_text'])
            else:
                 # Update background but keep original status foreground color (red/green)
                 label.config(bg=theme['bg_frame'])
        
        for btn in [self.hires_btn, self.weather_btn]:
            btn.config(bg=theme['btn_main'], fg=theme['btn_text'], activebackground=theme['bg_input'])
        
        self.analyze_button.config(bg=theme['btn_run'], fg=theme['btn_text'], activebackground=theme['bg_input'])
        self.toggle_button.config(bg=theme['bg_main'], fg=theme['fg_text'], activebackground=theme['bg_input'])

        self.result_text.config(bg=theme['bg_input'], fg=theme['fg_text'], insertbackground=theme['fg_text'])

    def toggle_mode(self):
        if self.is_dark_mode:
            self.current_theme = LIGHT_THEME
            self.is_dark_mode = False
            self.toggle_button.config(text="🌙 Dark Mode")
        else:
            self.current_theme = DARK_THEME
            self.is_dark_mode = True
            self.toggle_button.config(text="☀️ Light Mode")
            
        self.apply_theme(self.current_theme)
        
        if self.analyze_button['text'] == "3. Run Analysis":
             try:
                self.run_analysis_gui(update_charts_only=True)
             except Exception:
                pass 
                
    def run_analysis_gui(self, update_charts_only=False):
        if not update_charts_only:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "--- Analysis Started (Loading and processing data)... ---\n\n")
            self.result_text.config(state=tk.DISABLED)
            self.notebook.select(self.report_page) 
        
        try:
            if not update_charts_only:
                 self.analyze_button.config(state=tk.DISABLED, text="Running Analysis...")

            results = run_rms_analysis(self.hires_file, self.weather_file)
            
            create_charts(results, self.chart_frame, self.is_dark_mode)

            if not update_charts_only:
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

    def _format_results(self, results: Dict[str, Any]) -> str:
        report = "="*50 + "\n"
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

# --- 4. Main Execution (Placeholder files are now ignored) ---

if __name__ == "__main__":
    # We no longer need to pass the actual file names here, as the button starts disabled.
    # The initial placeholder text "Selected: None" will be displayed.
    HIRES_FILE = '' 
    WEATHER_FILE = ''

    root = tk.Tk()
    app = RMSAnalysisApp(root, HIRES_FILE, WEATHER_FILE)
    root.mainloop()