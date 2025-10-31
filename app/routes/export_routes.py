import io
import datetime as dt
from typing import Optional
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from fastapi import APIRouter, HTTPException, Header, Response
import firebase_admin
from firebase_admin import auth, firestore

router = APIRouter()

# Get Firestore client - will be set from main.py
db = None

def set_firestore_db(firestore_db):
    """Set the Firestore database instance from main.py"""
    global db
    db = firestore_db

# Removed admin verification - endpoints are now public

@router.get("/export/predictions")
def export_predictions():
    """Export all predictions as CSV"""
    
    if not db:
        raise HTTPException(status_code=500, detail="Firestore not initialized")
    
    try:
        # Fetch all predictions
        docs = db.collection('predictions').stream()
        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            doc_data['id'] = doc.id
            data.append(doc_data)
        
        if not data:
            raise HTTPException(status_code=404, detail="No predictions found")
        
        # Convert to DataFrame and CSV
        df = pd.DataFrame(data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        filename = f"predictions_export_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export/logs")
def export_logs():
    """Export all system logs as CSV"""
    
    if not db:
        raise HTTPException(status_code=500, detail="Firestore not initialized")
    
    try:
        # Fetch all system logs
        docs = db.collection('system_logs').stream()
        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            doc_data['id'] = doc.id
            data.append(doc_data)
        
        if not data:
            raise HTTPException(status_code=404, detail="No logs found")
        
        # Convert to DataFrame and CSV
        df = pd.DataFrame(data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        filename = f"logs_export_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export/report")
def export_report():
    """Generate enhanced monthly PDF report with town risk comparison and weekly trends"""
    
    if not db:
        raise HTTPException(status_code=500, detail="Firestore not initialized")
    
    try:
        now = dt.datetime.now()
        start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        week_ago = now - dt.timedelta(days=7)
        
        # Fetch current month predictions
        predictions_query = db.collection('predictions').where('timestamp', '>=', start_of_month).stream()
        predictions_data = [doc.to_dict() for doc in predictions_query]
        
        # Fetch last 7 days for trend analysis
        weekly_query = db.collection('predictions').where('timestamp', '>=', week_ago).stream()
        weekly_data = [doc.to_dict() for doc in weekly_query]
        
        if not predictions_data:
            raise HTTPException(status_code=404, detail="No data for current month")
        
        # Calculate statistics
        total_predictions = len(predictions_data)
        heatwave_events = sum(1 for p in predictions_data if p.get('alert', False))
        
        df = pd.DataFrame(predictions_data)
        weekly_df = pd.DataFrame(weekly_data) if weekly_data else pd.DataFrame()
        
        # Create charts
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Chart 1: Town Risk Comparison (Feature 1)
        if 'town' in df.columns and 'probability' in df.columns:
            latest_by_town = df.groupby('town')['probability'].last().sort_values(ascending=True)
            colors = ['red' if p >= 0.7 else 'orange' if p >= 0.4 else 'green' for p in latest_by_town.values]
            latest_by_town.plot(kind='barh', ax=ax1, color=colors)
            ax1.set_title('Current Town Risk Levels', fontweight='bold')
            ax1.set_xlabel('Heatwave Probability')
            ax1.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='High Risk')
            ax1.axvline(x=0.4, color='orange', linestyle='--', alpha=0.7, label='Moderate Risk')
            ax1.legend()
        
        # Chart 2: Weekly Trend Analysis (Feature 2)
        if not weekly_df.empty and 'date' in weekly_df.columns:
            weekly_df['date'] = pd.to_datetime(weekly_df['date'])
            daily_avg = weekly_df.groupby(weekly_df['date'].dt.date)['probability'].mean()
            daily_alerts = weekly_df.groupby(weekly_df['date'].dt.date)['alert'].sum()
            
            ax2_twin = ax2.twinx()
            daily_avg.plot(ax=ax2, color='blue', marker='o', label='Avg Probability')
            daily_alerts.plot(ax=ax2_twin, color='red', marker='s', label='Alert Count')
            ax2.set_title('7-Day Prediction Trends', fontweight='bold')
            ax2.set_ylabel('Average Probability', color='blue')
            ax2_twin.set_ylabel('Alert Count', color='red')
            ax2.tick_params(axis='x', rotation=45)
        
        # Chart 3: Daily Activity (existing)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            daily_counts = df.groupby(df['date'].dt.date).size()
            daily_counts.plot(kind='bar', ax=ax3, color='skyblue')
            ax3.set_title('Daily Prediction Activity')
            ax3.set_ylabel('Predictions Count')
            ax3.tick_params(axis='x', rotation=45)
        
        # Chart 4: Risk Distribution
        if 'probability' in df.columns:
            risk_bins = pd.cut(df['probability'], bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Moderate', 'High'])
            risk_counts = risk_bins.value_counts()
            colors_pie = ['green', 'orange', 'red']
            risk_counts.plot(kind='pie', ax=ax4, colors=colors_pie, autopct='%1.1f%%')
            ax4.set_title('Risk Level Distribution')
            ax4.set_ylabel('')
        
        plt.tight_layout()
        
        # Save charts to buffer
        chart_buffer = io.BytesIO()
        plt.savefig(chart_buffer, format='png', dpi=300, bbox_inches='tight')
        chart_buffer.seek(0)
        plt.close()
        
        # Create PDF
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph(f"Harara Heatwave Enhanced Report - {now.strftime('%B %Y')}", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Enhanced Statistics
        high_risk_towns = len([p for p in predictions_data if p.get('probability', 0) >= 0.7])
        avg_weekly_prob = weekly_df['probability'].mean() if not weekly_df.empty else 0
        
        stats_text = f"""
        <b>Monthly Statistics:</b><br/>
        • Total Predictions: {total_predictions}<br/>
        • Heatwave Events Detected: {heatwave_events}<br/>
        • Detection Rate: {(heatwave_events/total_predictions*100):.1f}%<br/>
        • High Risk Towns: {high_risk_towns}<br/>
        • Weekly Average Risk: {avg_weekly_prob:.2f}<br/>
        • Report Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}
        """
        stats = Paragraph(stats_text, styles['Normal'])
        story.append(stats)
        story.append(Spacer(1, 20))
        
        # Charts
        chart_image = Image(chart_buffer, width=7*inch, height=5*inch)
        story.append(chart_image)
        
        doc.build(story)
        pdf_content = pdf_buffer.getvalue()
        
        filename = f"harara_enhanced_report_{now.strftime('%Y%m')}.pdf"
        
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))