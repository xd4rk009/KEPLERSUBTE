import streamlit as st
import pandas as pd
import re
import uuid
import os
import json
import time
from io import BytesIO
import requests
from datetime import datetime
import numpy as np
import plotly.graph_objects as go

# IMPORTANTE: Se requiere instalar: pip install databricks-sql-connector plotly numpy openpyxl
from databricks import sql as databricks_sql 

# ============================================================
# CONFIGURACIÓN DE MODELOS (Referencia para visualización y UI)
# ============================================================
AVAILABLE_MODELS = [
    {"label": "FeedForward (Red Neuronal)", "type": "feedforward"},
    {"label": "Mixed (Red Mixta)", "type": "mixed"},
    {"label": "XGBoost", "type": "xgboost"},
    {"label": "Bayesian (Probabilístico)", "type": "bayesian"}
]

# ============================================================
# 1. CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="KEPLER",
    layout="wide",
    page_icon="-"
)

# ============================================================
# 2. GESTIÓN DE SESIÓN (LOGIN)
# ============================================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def login_screen():
    # --- CSS Y JS PARA FONDO DE PARTÍCULAS ---
    particles_html = """
    <style>
        #particles-js {
            position: fixed;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            z-index: -1;
            background: #0e1117;
        }
    </style>

    <canvas id="particles-js"></canvas>

    <script>
    const canvas = document.getElementById("particles-js");
    const ctx = canvas.getContext("2d");

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener("resize", resize);

    # Partículas muy simples
    const particles = Array.from({ length: 80 }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 2 + 1,
        speedX: (Math.random() - 0.5) * 0.6,
        speedY: (Math.random() - 0.5) * 0.6,
    }));

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        particles.forEach(p => {
            p.x += p.speedX;
            p.y += p.speedY;

            if (p.x < 0 || p.x > canvas.width) p.speedX *= -1;
            if (p.y < 0 || p.y > canvas.height) p.speedY *= -1;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(0, 200, 255, 0.8)";
            ctx.fill();
        });

        requestAnimationFrame(draw);
    }
    draw();
    </script>
    """

    st.components.v1.html(particles_html, height=0, width=0, scrolling=False)

    # --- FORMULARIO DE LOGIN (CENTRO) ---
    st.markdown("<h1 style='text-align: center; color: white;'>Acceso </h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>KEPLER</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Usuario", placeholder="admin")
            password = st.text_input("Contraseña", type="password", placeholder="admin")
            submit = st.form_submit_button("Ingresar al Sistema", type="primary", use_container_width=True)

            if submit:
                if username == "admin" and password == "admin":
                    st.session_state['logged_in'] = True
                    st.success("Acceso concedido. Redirigiendo...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos")

# ============================================================
# 3. APLICACIÓN PRINCIPAL
# ============================================================
def main_app():
    # BOTÓN DE SALIDA (LOGOUT)
    with st.sidebar:
        st.write(f"**Usuario:** Admin")
        if st.button("Cerrar Sesión", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()
        st.divider()

    # Configuración Databricks (Credenciales)
    DATABRICKS_TOKEN = "dapi57e006c218637b27ef24bbff623293d8"
    DATABRICKS_HOST = "https://dbc-55f9633a-48bc.cloud.databricks.com"
    DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/6b2bfefcc88e8ac0"

    # Rutas de almacenamiento de archivos (Files API)
    DATABRICKS_PATH = "/Volumes/kepler/raw/vol/guitarra1"
    DATABRICKS_PATH_RAW = "/Volumes/kepler/etl/landing-zone"   

    # Archivo local para persistencia del historial
    ARCHIVO_HISTORIAL = "historial_cargas_guitarra.json"

    # Columnas objetivo con patrones de búsqueda alternativos
    COLUMNAS_OBJETIVO = {
        'fecha': {
            'nombre_final': 'fecha',
            'patrones': ['fecha', 'date'],
            'es_fecha': True
        },
        'hora': {
            'nombre_final': 'hora',
            'patrones': ['hora', 'time'],
            'es_hora': True
        },
        'kg_de_explosivos_tronadura': {
            'nombre_final': 'kg_de_explosivos_tronadura',
            'patrones': ['kg_de_explosivos_tronadura', 'kg_explosivos_tronadura', 'tronadura.*kg.*explosivos', 'kg.*explosivos.*tronadura']
        },
        'kg_de_explosivos_destressing': {
            'nombre_final': 'kg_de_explosivos_destressing',
            'patrones': ['kg_de_explosivos_destressing', 'kg_explosivos_destressing', 'destressing.*kg.*explosivos', 'kg.*explosivos.*destressing']
        },
        'ucs_mpa': {
            'nombre_final': 'ucs_mpa',
            'patrones': ['ucs_mpa', 'ucs']
        },
        'modulo_de_young_gpa': {
            'nombre_final': 'modulo_de_young_gpa',
            'patrones': ['modulo_de_young', 'young']
        },
        'razon_poisson': {
            'nombre_final': 'razon_poisson',
            'patrones': ['razon_poisson', 'poisson']
        },
        'puntaje': {
            'nombre_final': 'puntaje',
            'patrones': ['puntaje']
        },
        'largo_real': {
            'nombre_final': 'largo_real',
            'patrones': ['largo_real', 'avance_real', 'avance.*real']
        },
        'area_teoricam2': {
            'nombre_final': 'area_teorica_m2',
            'patrones': ['area_teorica', 'area.*teorica'],
            'valor_default': 41.4
        },
        'volumen_excavado_m3': {
            'nombre_final': 'volumen_excavado_m3',
            'patrones': ['volumen_excavado', 'volumen.*excavado'],
            'calcular_de': ['area_teorica_m2', 'largo_real']
        }
    }

    # ============================================================
    # FUNCIONES AUXILIARES Y HISTORIAL
    # ============================================================

    def cargar_historial():
        """Carga el historial de archivos subidos desde un JSON local."""
        if os.path.exists(ARCHIVO_HISTORIAL):
            try:
                with open(ARCHIVO_HISTORIAL, 'r') as f:
                    return pd.DataFrame(json.load(f))
            except:
                return pd.DataFrame(columns=["uuid", "fecha", "hora", "archivo", "tipo", "modelos"])
        else:
            return pd.DataFrame(columns=["uuid", "fecha", "hora", "archivo", "tipo", "modelos"])

    def guardar_en_historial(uuid_val, nombre_archivo, tipo_carga="procesado", modelos_elegidos=None):
        """Guarda un nuevo registro en el historial JSON."""
        nuevo_registro = {
            "uuid": uuid_val,
            "fecha": datetime.now().strftime("%Y-%m-%d"),
            "hora": datetime.now().strftime("%H:%M:%S"),
            "archivo": nombre_archivo,
            "tipo": tipo_carga,
            "modelos": str(modelos_elegidos) if modelos_elegidos else "N/A"
        }
        
        lista_actual = []
        if os.path.exists(ARCHIVO_HISTORIAL):
            try:
                with open(ARCHIVO_HISTORIAL, 'r') as f:
                    lista_actual = json.load(f)
            except:
                lista_actual = []
        
        # Insertar al inicio (más reciente primero)
        lista_actual.insert(0, nuevo_registro)
        
        with open(ARCHIVO_HISTORIAL, 'w') as f:
            json.dump(lista_actual, f, indent=4)

    def graficar_con_analisis(df, col_x, col_y, titulo_y, thresholds=[0, 0.6, 1.2, 1.8], log_y=False, convertir_mw=False, unique_key="chart"):
        """
        Función genérica para graficar datos con Plotly, incluyendo suavizado,
        líneas de umbral y conteos estadísticos.
        """
        if df.empty or col_x not in df.columns or col_y not in df.columns:
            st.warning(f"No se pueden graficar los datos. Faltan columnas: {col_x}, {col_y}")
            return

        # Preparar datos
        df_chart = df.copy()
        
        # Ordenar por X
        df_chart = df_chart.sort_values(by=col_x)

        # Conversión a Mw EQ si se solicita
        label_y = titulo_y
        
        # Calcular y_final BASE (antes de cualquier filtro visual)
        if convertir_mw:
            # Filtrar valores <= 0 para evitar error matemático en log10
            df_chart = df_chart[df_chart[col_y] > 0].copy()
            
            # Fórmula: Mw = 2/3 * log10(prediction) - 6.21
            df_chart['y_final'] = (2/3) * np.log10(df_chart[col_y]) - 6.21
            label_y = "Mw EQ (Magnitud)"
        else:
            df_chart['y_final'] = df_chart[col_y]

        # Panel de Controles Gráficos
        c1, c2 = st.columns(2)
        with c1:
            # Usamos unique_key para evitar conflictos de ID en Streamlit
            usar_suavizado = st.checkbox("Aplicar Media Móvil (Suavizado)", value=False, key=f"chk_smooth_{unique_key}")
        with c2:
            window_size = st.slider("Ventana de suavizado", 2, 50, 5, key=f"sld_smooth_{unique_key}", disabled=not usar_suavizado)

        # ---------------------------------------------------------
        # CÁLCULO DE TABLA DE UMBRALES (Sobre la variable y_final)
        # ---------------------------------------------------------
        conteos = []
        total_datos = len(df_chart)
        
        for t in thresholds:
            # Contamos sobre y_final (sea Raw o Mw EQ)
            count = len(df_chart[df_chart['y_final'] > t])
            pct = (count / total_datos * 100) if total_datos > 0 else 0
            conteos.append({
                "Umbral": f"> {t}", 
                "Cantidad": count, 
                "Porcentaje": f"{pct:.1f}%",
                "Variable": label_y # Para que sea explícito qué estamos contando
            })
        
        df_conteos = pd.DataFrame(conteos)

        # Mostrar conteos
        st.markdown("##### Análisis de Umbrales")
        st.caption(f"Conteos calculados sobre: **{label_y}**")
        st.table(df_conteos)

        # ---------------------------------------------------------
        # CREAR GRÁFICA
        # ---------------------------------------------------------
        fig = go.Figure()

        # 1. Puntos originales (Scatter)
        fig.add_trace(go.Scatter(
            x=df_chart[col_x], 
            y=df_chart['y_final'], 
            mode='markers',
            name='Datos Reales',
            marker=dict(size=6, opacity=0.6, color='#3366CC')
        ))

        # 2. Línea de tendencia suavizada
        if usar_suavizado:
            # Calcular media móvil sobre y_final
            y_smooth = df_chart['y_final'].rolling(window=window_size, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=df_chart[col_x],
                y=y_smooth,
                mode='lines',
                name=f'Media Móvil ({window_size})',
                line=dict(color='red', width=2)
            ))

        # 3. Líneas de Umbral
        colores_umbral = ['green', 'orange', 'darkred', 'purple']
        for i, t in enumerate(thresholds):
            color = colores_umbral[i % len(colores_umbral)]
            fig.add_hline(y=t, line_dash="dash", line_color=color, annotation_text=f"Limit {t}", annotation_position="bottom right")

        # Configuración del Layout
        layout_args = dict(
            title=f"Gráfico: {col_x} vs {label_y}",
            xaxis_title=col_x,
            yaxis_title=label_y,
            template="plotly_white",
            height=500,
            hovermode="x unified"
        )

        # Log scale solo si NO estamos ya en Mw (porque Mw ya es logarítmico)
        if log_y and not convertir_mw: 
            layout_args['yaxis_type'] = "log"

        fig.update_layout(**layout_args)
        st.plotly_chart(fig, use_container_width=True)


    # ============================================================
    # FUNCIONES DE PROCESAMIENTO
    # ============================================================

    def ejecutar_sql_databricks(query, host, token, http_path):
        """
        Ejecuta una consulta SQL en Databricks usando el SQL Connector
        Retorna un DataFrame de Pandas.
        """
        try:
            if not host or not token or not http_path:
                return None, "Faltan credenciales (Host, Token o HTTP Path)"

            server_hostname = host.replace("https://", "").replace("http://", "").rstrip("/")

            with databricks_sql.connect(
                server_hostname=server_hostname,
                http_path=http_path,
                access_token=token
            ) as connection:
                
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    result = cursor.fetchall()
                    
                    if cursor.description:
                        column_names = [desc[0] for desc in cursor.description]
                        df = pd.DataFrame(result, columns=column_names)
                        return df, None
                    else:
                        return pd.DataFrame(), None # Para queries que no retornan datos

        except Exception as e:
            return None, f"Error SQL: {str(e)}"

    def limpiar_nombre_columna(col_tuple):
        """Convierte tuplas de MultiIndex a nombres estilo snake_case"""
        nivel_0 = str(col_tuple[0]) if len(col_tuple) > 0 else ''
        nivel_1 = str(col_tuple[1]) if len(col_tuple) > 1 else ''
        
        def limpiar_texto(texto):
            texto = texto.lower()
            texto = texto.replace('\n', ' ')
            texto = re.sub(r'[°º]', '', texto)
            texto = re.sub(r'[()]', '', texto)
            texto = re.sub(r'[áàâã]', 'a', texto)
            texto = re.sub(r'[éèê]', 'e', texto)
            texto = re.sub(r'[íìî]', 'i', texto)
            texto = re.sub(r'[óòô]', 'o', texto)
            texto = re.sub(r'[úùû]', 'u', texto)
            texto = re.sub(r'ñ', 'n', texto)
            texto = re.sub(r'[\s\-/]+', '_', texto)
            texto = re.sub(r'[^a-z0-9_]', '', texto)
            texto = re.sub(r'_+', '_', texto)
            return texto.strip('_')
        
        nivel_0_limpio = limpiar_texto(nivel_0)
        nivel_1_limpio = limpiar_texto(nivel_1)
        
        # CASOS ESPECIALES PARA EXPLOSIVOS
        if nivel_1_limpio and ('kg' in nivel_1_limpio and 'explosivos' in nivel_1_limpio):
            if 'destressing' in nivel_0_limpio:
                return f"kg_explosivos_destressing"
            elif 'tronadura' in nivel_0_limpio:
                return f"kg_explosivos_tronadura"
            elif nivel_0_limpio and 'unnamed' not in nivel_0_limpio:
                return f"{nivel_0_limpio}_{nivel_1_limpio}"
        
        if 'destressing' in nivel_0_limpio and nivel_1_limpio and 'unnamed' not in nivel_1_limpio:
            if not ('kg' in nivel_1_limpio and 'explosivos' in nivel_1_limpio):
                return f"{nivel_1_limpio}_{nivel_0_limpio}"
        
        partes = []
        if nivel_0_limpio and 'unnamed' not in nivel_0_limpio:
            partes.append(nivel_0_limpio)
        if nivel_1_limpio and 'unnamed' not in nivel_1_limpio:
            partes.append(nivel_1_limpio)
        
        return '_'.join(partes) if partes else 'columna_sin_nombre'


    def hacer_nombres_unicos(columnas):
        """Agrega sufijos numéricos a nombres duplicados"""
        columnas_unicas = []
        contador = {}
        duplicados = [col for col in set(columnas) if columnas.count(col) > 1]
        
        for col in columnas:
            if col in duplicados:
                contador[col] = contador.get(col, 0) + 1
                columnas_unicas.append(f"{col}_{contador[col]}")
            else:
                columnas_unicas.append(col)
        
        return columnas_unicas, duplicados


    def buscar_columnas(df, columnas_dict):
        """Busca columnas usando patrones alternativos y regex"""
        columnas_encontradas = {}
        
        for clave, config in columnas_dict.items():
            nombre_final = config['nombre_final']
            patrones = config['patrones']
            
            for patron in patrones:
                coincidencias = [col for col in df.columns 
                                 if re.search(patron, col.lower())]
                
                if coincidencias:
                    columnas_encontradas[clave] = {
                        'columna_original': coincidencias[0],
                        'nombre_final': nombre_final,
                        'patron_usado': patron,
                        'valor_default': config.get('valor_default', None),
                        'calcular_de': config.get('calcular_de', None),
                        'es_fecha': config.get('es_fecha', False),
                        'es_hora': config.get('es_hora', False)
                    }
                    break
        
        return columnas_encontradas


    def procesar_fecha_hora(df_filtrado, columnas_encontradas):
        """
        Combina las columnas Fecha y Hora en una columna datetime.
        """
        tiene_fecha = 'fecha' in columnas_encontradas
        tiene_hora = 'hora' in columnas_encontradas
        
        if not (tiene_fecha and tiene_hora):
            return df_filtrado, None
        
        try:
            df_temp = df_filtrado.copy()
            
            # Convertir a string y limpiar
            df_temp['fecha_limpia'] = df_filtrado['fecha'].astype(str).str.strip()
            df_temp['hora_limpia'] = df_filtrado['hora'].astype(str).str.strip()
            
            # Filtrar valores no validos
            mascara_validos = (
                (df_temp['fecha_limpia'] != 'nan') & 
                (df_temp['fecha_limpia'] != 'None') & 
                (df_temp['fecha_limpia'] != '') &
                (df_temp['hora_limpia'] != 'nan') & 
                (df_temp['hora_limpia'] != 'None') & 
                (df_temp['hora_limpia'] != '')
            )
            
            datetime_col = pd.Series([pd.NaT] * len(df_temp), index=df_temp.index)
            
            if mascara_validos.any():
                for idx in df_temp[mascara_validos].index:
                    fecha_str = df_temp.loc[idx, 'fecha_limpia']
                    hora_str = df_temp.loc[idx, 'hora_limpia']
                    
                    hora_str = hora_str.replace('\n', ' ').replace('\r', ' ')
                    hora_str = re.sub(r'\s+', ' ', hora_str).strip()
                    hora_str = re.sub(r'^(TA|TB)\s*', '', hora_str, flags=re.IGNORECASE)
                    hora_str = re.sub(r'^:\s*', '', hora_str)
                    
                    if (' ' in fecha_str and ':' in fecha_str and ' ' in hora_str and ':' in hora_str):
                        if fecha_str == hora_str:
                            try:
                                datetime_col[idx] = pd.to_datetime(fecha_str, errors='coerce')
                                continue
                            except:
                                pass
                        else:
                            try:
                                datetime_col[idx] = pd.to_datetime(hora_str, errors='coerce')
                                continue
                            except:
                                pass
                    
                    if ' ' in fecha_str and ':' in fecha_str and ' ' not in hora_str:
                        fecha_parte = fecha_str.split(' ')[0]
                        datetime_str = f"{fecha_parte} {hora_str}"
                        try:
                            datetime_col[idx] = pd.to_datetime(datetime_str, errors='coerce')
                            continue
                        except:
                            pass
                    
                    datetime_str = f"{fecha_str} {hora_str}"
                    
                    for fmt in ['%d-%m-%Y %H:%M:%S', '%d-%m-%Y %H:%M']:
                        try:
                            datetime_col[idx] = pd.to_datetime(datetime_str, format=fmt, errors='raise')
                            break
                        except:
                            continue
                    
                    if pd.isna(datetime_col[idx]):
                        try:
                            datetime_col[idx] = pd.to_datetime(datetime_str, errors='coerce', dayfirst=True)
                        except:
                            pass
            
            df_resultado = df_filtrado.copy()
            df_resultado.insert(0, 'datetime', datetime_col)
            df_resultado = df_resultado.drop(columns=['fecha', 'hora'])
            
            exitosas = datetime_col.notna().sum()
            fallidas = datetime_col.isna().sum()
            
            ejemplos_fallidos = []
            if fallidas > 0:
                indices_fallidos = datetime_col[datetime_col.isna()].index[:5]
                for idx in indices_fallidos:
                    ejemplos_fallidos.append({
                        'fecha': df_temp.loc[idx, 'fecha_limpia'],
                        'hora': df_temp.loc[idx, 'hora_limpia'],
                        'combinado': f"{df_temp.loc[idx, 'fecha_limpia']} {df_temp.loc[idx, 'hora_limpia']}"
                    })
            
            return df_resultado, {
                'exitosas': exitosas,
                'fallidas': fallidas,
                'total': len(datetime_col),
                'ejemplos_fallidos': ejemplos_fallidos
            }
            
        except Exception as e:
            st.warning(f"No se pudo procesar fecha/hora: {str(e)}")
            return df_filtrado, None

    def convertir_df_a_excel(df):
        """Convierte DataFrame a bytes para descarga"""
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        return output.getvalue()


    def guardar_en_databricks(df, token, host, path):
        """
        Guarda el DataFrame en Databricks usando la Files API.
        RETORNA: (Exito: bool, Mensaje: str, UUID: str)
        """
        try:
            if not host:
                return False, "Por favor ingresa el Host de Databricks", None
            
            # Generar UUID4 y Timestamp
            unique_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"{unique_id}_procesado_{timestamp}.xlsx"
            full_path = f"{path.rstrip('/')}/{filename}"
            
            excel_data = convertir_df_a_excel(df)
            
            url = f"{host.rstrip('/')}/api/2.0/fs/files{full_path}"
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/octet-stream"
            }
            
            response = requests.put(
                url,
                headers=headers,
                data=excel_data
            )
            
            if response.status_code in [200, 201, 204]:
                return True, full_path, unique_id
            else:
                return False, f"Error {response.status_code}: {response.text}", None
                
        except Exception as e:
            return False, str(e), None


    #
    def guardar_archivo_crudo_en_databricks(file_bytes, original_filename, token, host, path, selected_model_types=None):
        """
        Guarda el archivo crudo en Databricks y un archivo JSON de metadatos asociado
        CON EL FORMATO ESPECÍFICO SOLICITADO.
        
        RETORNA: (Exito: bool, Mensaje: str, UUID: str)
        """
        try:
            if not host:
                return False, "Por favor ingresa el Host de Databricks", None
            
            # Generar UUID4 y Timestamp
            unique_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # --- 1. SUBIDA DEL ARCHIVO EXCEL ---
            # Nota: El JSON pide "blast_filename", usaremos el nombre generado en databricks
            filename_excel = f"{unique_id}_crudo_{timestamp}_{original_filename}"
            full_path_excel = f"{path.rstrip('/')}/{filename_excel}"
            
            url_excel = f"{host.rstrip('/')}/api/2.0/fs/files{full_path_excel}"
            
            headers_excel = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/octet-stream"
            }
            
            response_excel = requests.put(url_excel, headers=headers_excel, data=file_bytes)
            
            # --- 2. CREACIÓN Y SUBIDA DEL METADATA JSON ---
            if response_excel.status_code in [200, 201, 204]:
                try:
                    # Construir lista de modelos para el JSON
                    # Formato: [ { "type": "feedforward" }, { "type": "mixed" } ... ]
                    requested_models_list = []
                    if selected_model_types:
                        for m_type in selected_model_types:
                            requested_models_list.append({"type": m_type})
                    
                    # Crear diccionario de metadatos CON EL FORMATO EXACTO SOLICITADO
                    metadata = {
                        "request_id": unique_id,
                        "ts_uploaded_utc": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                        "blast_filename": filename_excel,
                        "requested_models": requested_models_list
                    }
                    
                    # Convertir a bytes JSON
                    json_bytes = json.dumps(metadata, indent=4).encode('utf-8')
                    
                    # Nombre solicitado: UUID_metadata.json
                    filename_json = f"{unique_id}_metadata.json"
                    full_path_json = f"{path.rstrip('/')}/{filename_json}"
                    url_json = f"{host.rstrip('/')}/api/2.0/fs/files{full_path_json}"
                    
                    headers_json = {
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"
                    }
                    
                    # Subir JSON
                    requests.put(url_json, headers=headers_json, data=json_bytes)
                    
                    # Retornamos éxito indicando que se subió el Excel (el JSON es acompañante)
                    return True, full_path_excel, unique_id

                except Exception as e_json:
                    # Si falla el JSON pero el Excel subió, retornamos éxito pero con advertencia en consola
                    print(f"Advertencia: Excel subido pero falló metadata JSON: {e_json}")
                    return True, full_path_excel, unique_id
            else:
                return False, f"Error al subir Excel {response_excel.status_code}: {response_excel.text}", None
                
        except Exception as e:
            return False, str(e), None

    # ============================================================
    # LÓGICA DE PROCESAMIENTO SÍSMICO (NUEVO)
    # ============================================================
    def procesar_sismicidad_real(df_sismica):
        """
        Procesa el archivo CSV de sismicidad para agrupar Mo por fecha.
        Formula Mo: Mo = 10 ^ (1.5 * (LocalMagnitude + 6.21))
        Formula Mw_EQ: (2/3) * log10(Sum_Mo) - 6.21
        """
        try:
            # 1. Limpieza de columnas (eliminar espacios y #)
            df_sismica.columns = [c.strip().replace('#', '') for c in df_sismica.columns]
            
            # Buscar columnas clave
            col_date = next((c for c in df_sismica.columns if 'Date' in c), None)
            col_mag = next((c for c in df_sismica.columns if 'Magnitude' in c), None)
            
            if not col_date or not col_mag:
                return None, "No se encontraron columnas 'Date' o 'Magnitude' en el CSV."

            # 2. Conversión a Datetime (Solo Fecha para agrupar)
            # Asumimos formato YYYY/MM/DD o YYYY-MM-DD
            df_sismica['fecha_grupo'] = pd.to_datetime(df_sismica[col_date]).dt.date
            
            # 3. Calcular Momento Sísmico (Mo) por evento
            # Formula inversa: Local Mag = (2/3) * log10(Mo) - 6.21
            # Despeje: log10(Mo) = (Local Mag + 6.21) * 1.5
            # Mo = 10 ^ ((Local Mag + 6.21) * 1.5)
            
            # Asegurar numérico
            df_sismica[col_mag] = pd.to_numeric(df_sismica[col_mag], errors='coerce')
            df_sismica = df_sismica.dropna(subset=[col_mag])
            
            df_sismica['Mo_calc'] = np.power(10, (1.5 * (df_sismica[col_mag] + 6.21)))
            
            # 4. Agrupar por Fecha (n_disparo lógico) y Sumar Mo
            df_grouped = df_sismica.groupby('fecha_grupo')['Mo_calc'].sum().reset_index()
            
            # 5. Calcular Mw EQ grupal
            # Formula: Mw = (2/3) * log10(Sum_Mo) - 6.21
            df_grouped['Mw_EQ_Real'] = (2/3) * np.log10(df_grouped['Mo_calc']) - 6.21
            
            return df_grouped, None
            
        except Exception as e:
            return None, str(e)


    def extraer_mapeo_fecha_disparo(file_excel):
        """
        Lee el Excel Original para crear un diccionario {Fecha: N_Disparo}
        """
        try:
            # Leer Excel con header complejo si es necesario, o intentar detección automática
            # Asumimos que es el mismo formato que en la Tab 1 (header=[0,1])
            df = pd.read_excel(file_excel, header=[0, 1])
            
            # Limpiar columnas para encontrar 'fecha' y 'n_disparo'
            # Usamos funciones auxiliares existentes pero simplificadas para este propósito especifico
            columnas_planas = [limpiar_nombre_columna(c) for c in df.columns]
            df.columns = columnas_planas
            
            # Buscar columnas necesarias
            # n_disparo a veces se llama diferente, buscamos 'disparo'
            col_disparo = next((c for c in df.columns if 'disparo' in c.lower() and 'n' in c.lower()), None)
            
            # fecha
            col_fecha = next((c for c in df.columns if 'fecha' in c.lower()), None)
            
            if not col_disparo or not col_fecha:
                return None, f"No se encontraron columnas 'N° Disparo' o 'Fecha' en el Excel de Mapeo. Cols: {df.columns.tolist()}"
                
            # Crear sub-dataframe limpio
            df_map = df[[col_fecha, col_disparo]].copy()
            df_map.columns = ['fecha_raw', 'n_disparo_map']
            
            # Convertir fecha a Date object
            df_map['fecha_obj'] = pd.to_datetime(df_map['fecha_raw'], errors='coerce').dt.date
            
            # Eliminar filas sin fecha o disparo
            df_map = df_map.dropna(subset=['fecha_obj', 'n_disparo_map'])
            
            # Asegurar n_disparo numerico
            df_map['n_disparo_map'] = pd.to_numeric(df_map['n_disparo_map'], errors='coerce')
            
            return df_map[['fecha_obj', 'n_disparo_map']], None

        except Exception as e:
            return None, str(e)


    # ============================================================
    # INTERFAZ STREAMLIT
    # ============================================================
    st.title("KEPLER")
    st.markdown("---")

    # TABS PRINCIPALES (MODIFICADO: SE ELIMINÓ VER TABLAS SQL)
    tab_procesar, tab_status = st.tabs([
        "Procesar Nuevo Archivo", 
        "Historial y estado UUID"
    ])

    # ============================================================
    # TAB 1: PROCESAMIENTO DE ARCHIVOS
    # ============================================================
    with tab_procesar:
        # ---------------------------------------------------------------------
        # NUEVO: Mostrar variables requeridas de forma compacta (st.info)
        # ---------------------------------------------------------------------
        cols_esperadas = [c['nombre_final'] for c in COLUMNAS_OBJETIVO.values()]
        st.info(f"**Variables requeridas en el Excel:** {', '.join(cols_esperadas)}")
        
        uploaded_file = st.file_uploader(
            "Cargar archivo Excel con MultiIndex",
            type=['xlsx', 'xls'],
            help="Selecciona el archivo Informacion_Guitarra_TC_P0_Zublin_312"
        )

        if uploaded_file is not None:
            
            file_bytes = uploaded_file.getvalue()
            original_filename = uploaded_file.name
            
            try:
                # Cargar datos
                df = pd.read_excel(uploaded_file, header=[0, 1])
                
                # Limpiar nombres
                columnas_limpias = [limpiar_nombre_columna(col) for col in df.columns]
                columnas_unicas, duplicados = hacer_nombres_unicos(columnas_limpias)
                df.columns = columnas_unicas
                
                # Buscar columnas objetivo
                columnas_encontradas = buscar_columnas(df, COLUMNAS_OBJETIVO)
                
                # Crear DataFrame filtrado
                columnas_originales = [info['columna_original'] for info in columnas_encontradas.values()]
                df_filtrado = df[columnas_originales].copy()
                
                # Renombrar columnas
                rename_dict = {info['columna_original']: info['nombre_final'] 
                            for info in columnas_encontradas.values()}
                df_filtrado.rename(columns=rename_dict, inplace=True)
                
                # 1. Procesar Fecha y Hora
                df_filtrado, resultado_datetime = procesar_fecha_hora(df_filtrado, columnas_encontradas)
                
                # 2. Si falta area_teorica_m2
                if 'area_teorica_m2' not in df_filtrado.columns:
                    config_area = COLUMNAS_OBJETIVO['area_teoricam2']
                    if 'valor_default' in config_area:
                        df_filtrado['area_teorica_m2'] = config_area['valor_default']
                        st.info(f"Columna 'area_teorica_m2' no encontrada. Se asigno valor por defecto: {config_area['valor_default']}")
                
                # 3. Si falta volumen_excavado_m3
                if 'volumen_excavado_m3' not in df_filtrado.columns:
                    config_volumen = COLUMNAS_OBJETIVO['volumen_excavado_m3']
                    columnas_necesarias = config_volumen.get('calcular_de', [])
                    
                    if all(col in df_filtrado.columns for col in columnas_necesarias):
                        df_filtrado['volumen_excavado_m3'] = (
                            df_filtrado['area_teorica_m2'] * df_filtrado['largo_real']
                        )
                        st.info(f"Columna 'volumen_excavado_m3' calculada como: area_teorica_m2 x largo_real")
                
                # Metricas principales
                st.subheader("Resultados del Procesamiento")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Filas totales", f"{len(df_filtrado):,}")
                with col2:
                    st.metric("Columnas finales", len(df_filtrado.columns))
                with col3:
                    st.metric("Columnas encontradas", f"{len(columnas_encontradas)}/{len(COLUMNAS_OBJETIVO)}")
                with col4:
                    st.metric("Valores nulos", f"{df_filtrado.isnull().sum().sum():,}")
                
                if resultado_datetime:
                    st.success(f"Columna 'datetime' creada exitosamente: {resultado_datetime['exitosas']} registros convertidos")
                    if resultado_datetime['fallidas'] > 0:
                        st.warning(f"{resultado_datetime['fallidas']} registros no pudieron convertirse y quedaran como NaT")
                        
                        if resultado_datetime.get('ejemplos_fallidos'):
                            with st.expander("Ver ejemplos de registros fallidos"):
                                st.write("Primeros 5 registros que no pudieron convertirse:")
                                for i, ejemplo in enumerate(resultado_datetime['ejemplos_fallidos'], 1):
                                    st.text(f"{i}. Fecha: '{ejemplo['fecha']}' | Hora: '{ejemplo['hora']}' | Combinado: '{ejemplo['combinado']}'")
                
                st.markdown("---")
                
                # -------------------------------------------------------------------
                # NUEVO: Mostrar estado de columnas COMPACTO (Ahorro de espacio)
                # -------------------------------------------------------------------
                
                found_list = []
                missing_list = []
                
                for clave, config in COLUMNAS_OBJETIVO.items():
                    nombre_final = config['nombre_final']
                    
                    if resultado_datetime and (config.get('es_fecha') or config.get('es_hora')):
                        continue
                    
                    if clave in columnas_encontradas:
                        found_list.append(nombre_final)
                    elif nombre_final in df_filtrado.columns:
                        if 'valor_default' in config:
                            found_list.append(f"{nombre_final} (Default)")
                        elif 'calcular_de' in config:
                            found_list.append(f"{nombre_final} (Calc)")
                    else:
                        missing_list.append(nombre_final)
                
                if found_list:
                    st.success(f"**Variables Encontradas:** {', '.join(found_list)}")
                
                if missing_list:
                    st.error(f"**Variables Faltantes:** {', '.join(missing_list)}")
                
                st.markdown("---")
                
                # ============================================================
                # SELECCIÓN DE MODELOS (NUEVA UBICACIÓN)
                # ============================================================
                st.subheader("Configuración de Modelos")
                st.write("Selecciona los modelos que deseas ejecutar con este archivo:")
                
                # Crear opciones para el multiselect
                opciones_display = [m["label"] for m in AVAILABLE_MODELS]
                
                # Multiselect por defecto todos seleccionados
                seleccionados_labels = st.multiselect(
                    "Modelos a ejecutar", 
                    options=opciones_display,
                    default=opciones_display
                )
                
                # Mapear de Label -> Type para el JSON
                selected_model_types = []
                for label in seleccionados_labels:
                    for m in AVAILABLE_MODELS:
                        if m["label"] == label:
                            selected_model_types.append(m["type"])
                            break
                
                st.markdown("---")

                # Opciones de descarga/subida
                st.subheader("Opciones de Exportacion")
                
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                
                with col_btn1:
                    excel_data = convertir_df_a_excel(df_filtrado)
                    st.download_button(
                        label="Descargar Excel Procesado",
                        data=excel_data,
                        file_name="datos_guitarra_procesado.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                with col_btn2:
                    if st.button("Subir Procesado a Databricks", use_container_width=True):
                        with st.spinner("Subiendo archivo procesado..."):
                            exito, mensaje, uuid_gen = guardar_en_databricks(
                                df_filtrado, 
                                DATABRICKS_TOKEN, 
                                DATABRICKS_HOST,
                                DATABRICKS_PATH
                            )
                            
                            if exito:
                                st.success(f"Archivo guardado. UUID: `{uuid_gen}`")
                                st.caption(f"Ruta: {mensaje}")
                                # GUARDAR EN HISTORIAL
                                guardar_en_historial(uuid_gen, original_filename, "Procesado")
                            else:
                                st.error(f"Error: {mensaje}")
                
                with col_btn3:
                    # Botón principal para ejecutar pipeline
                    if st.button("Subir Archivo Crudo a Databricks", use_container_width=True):
                        if not selected_model_types:
                            st.error("Debes seleccionar al menos un modelo.")
                        else:
                            with st.spinner("Subiendo archivo crudo y configuración..."):
                                exito, mensaje, uuid_gen = guardar_archivo_crudo_en_databricks(
                                    file_bytes,
                                    original_filename,
                                    DATABRICKS_TOKEN, 
                                    DATABRICKS_HOST,
                                    DATABRICKS_PATH_RAW,
                                    selected_model_types=selected_model_types
                                )
                                
                                if exito:
                                    st.success(f"Archivo Crudo guardado. UUID: `{uuid_gen}`")
                                    st.caption(f"Ruta: {mensaje}")
                                    st.info(f"Modelos solicitados: {', '.join(selected_model_types)}")
                                    # GUARDAR EN HISTORIAL
                                    guardar_en_historial(uuid_gen, original_filename, "Crudo", selected_model_types)
                                else:
                                    st.error(f"Error: {mensaje}")
                                
            except Exception as e:
                st.error(f"Error al procesar el archivo: {str(e)}")

        else:
            st.info("Por favor, carga un archivo Excel para comenzar el procesamiento en esta pestaña.")

    # ============================================================
    # TAB 2: HISTORIAL Y ESTADO (MODIFICADO CON CALIBRACIÓN)
    # ============================================================
    with tab_status:
        st.header("Historial de Cargas y Monitor de Estado")
        st.markdown("Aquí puedes ver el historial de archivos subidos y rastrear su estado de procesamiento en Databricks usando su UUID.")
        
        # ----------------------------------------------------
        # SECCIÓN 1: HISTORIAL (ARRIBA)
        # ----------------------------------------------------
        df_history = cargar_historial()
        
        st.subheader("Archivos Subidos (Historial)")
        st.caption("Este historial persiste aunque cierres la app.")
        
        if not df_history.empty:
            st.dataframe(df_history, use_container_width=True, height=300)
        else:
            st.info("No hay historial de cargas aún.")

        st.divider() # LÍNEA DIVISORIA

        # ----------------------------------------------------
        # SECCIÓN 2: CONSULTA DE ESTADO (ABAJO)
        # ----------------------------------------------------
        st.subheader("Consultar Estado (Multi-Modelos)")
        
        # Selector de UUID del historial
        opciones_uuid = []
        if not df_history.empty:
            opciones_uuid = df_history['uuid'].tolist()
        
        uuid_input = st.selectbox("Seleccionar UUID del historial:", opciones_uuid)
        
        # Campo de texto por si quieren pegar uno manual
        uuid_manual = st.text_input("O pegar UUID manual:", value=uuid_input if uuid_input else "")
        
        uuid_final = uuid_manual if uuid_manual else uuid_input
        
        # ------------------------------------------------------------
        # MODIFICACIÓN: LÓGICA DE MONITOREO AUTOMÁTICO (LOOP)
        # ------------------------------------------------------------
        if st.button("Iniciar Monitoreo de Estado", disabled=not uuid_final):
            st.session_state['uuid_current'] = uuid_final
            
            # Contenedores vacíos para actualizar la UI en cada iteración
            status_container = st.empty()
            progress_bar = st.progress(0)
            
            st.markdown(f"**Monitoreando UUID:** `{uuid_final}`")
            
            # Bucle de monitoreo
            monitoreando = True
            while monitoreando:
                query_status = f"SELECT request_status FROM kepler.etl.request_status WHERE request_id = '{uuid_final}'"
                
                # Ejecutar query
                df_status, error_status = ejecutar_sql_databricks(
                    query_status, DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH
                )
                
                if error_status:
                    status_container.error(f"Error de conexión: {error_status}")
                    monitoreando = False
                    break
                
                if df_status is None or df_status.empty:
                    status_container.warning("UUID no encontrado aún en la base de datos. Esperando...")
                    time.sleep(3)
                    continue
                
                # Obtener código de estado
                estado_actual = df_status.iloc[0,0]
                st.session_state['status_code'] = estado_actual
                
                if estado_actual == 0:
                    status_container.info("Estado: Inicializando procesamiento en Databricks...")
                    progress_bar.progress(10)
                    time.sleep(3)
                    
                elif estado_actual == 1:
                    status_container.info("Estado: Procesando datos (Ejecutando modelos)...")
                    progress_bar.progress(50)
                    time.sleep(3)
                    
                elif estado_actual == 2:
                    status_container.success("Estado: Procesamiento Finalizado Exitosamente.")
                    progress_bar.progress(100)
                    monitoreando = False # Salir del bucle
                    
                    # Descargar resultados automáticamente al finalizar
                    query_result = f"SELECT * FROM kepler.etl.request_result WHERE request_id = '{uuid_final}'"
                    with st.spinner("Descargando resultados finales..."):
                        df_res, err_res = ejecutar_sql_databricks(
                            query_result, DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH
                        )
                        if df_res is not None:
                            st.session_state['status_result'] = df_res
                        else:
                            st.session_state['status_result'] = pd.DataFrame()
                    
                elif estado_actual == 3:
                    status_container.error("Estado: Error en el procesamiento.")
                    progress_bar.progress(100)
                    monitoreando = False
                    
                else:
                    status_container.info(f"Estado Desconocido: {estado_actual}")
                    monitoreando = False

        # RENDERIZADO BASADO EN SESSION STATE (Para mostrar resultados después del loop)
        if 'status_code' in st.session_state:
            estado = st.session_state['status_code']
            
            # Si ya terminó (estado 2) y tenemos resultados, mostramos todo el panel de análisis
            if estado == 2:
                if 'status_result' in st.session_state and not st.session_state['status_result'].empty:
                    df_res = st.session_state['status_result']
                    
                    st.divider()
                    
                    # -------------------------------------------------------
                    # LOGICA DE VISUALIZACIÓN MULTI-MODELO FILTRADA
                    # -------------------------------------------------------
                    
                    # Asegurar tipos de datos numéricos para filtrado y graficado correcto
                    if 'n_disparo' in df_res.columns and 'estimacion' in df_res.columns and 'modelo' in df_res.columns:
                        try:
                            df_res['n_disparo'] = pd.to_numeric(df_res['n_disparo'])
                            df_res['estimacion'] = pd.to_numeric(df_res['estimacion'])
                        except:
                            pass
                        
                        # --- 1. SELECTOR DE MODELOS (REQ USUARIO) ---
                        modelos_totales = sorted(df_res['modelo'].unique().tolist())
                        st.markdown("**Resultados Disponibles:**")
                        
                        # Selector para filtrar qué ver (arriba de la tabla)
                        modelos_seleccionados = st.multiselect(
                            "Seleccionar Modelos a Visualizar:",
                            options=modelos_totales,
                            default=modelos_totales
                        )
                        
                        if not modelos_seleccionados:
                            st.warning("Por favor selecciona al menos un modelo para visualizar.")
                        else:
                            # --- 2. FILTRADO DE DATOS (Afecta Tabla, Gráfico y Resumen) ---
                            df_filtrada_vista = df_res[df_res['modelo'].isin(modelos_seleccionados)].copy()
                            
                            # --- 3. TABLA (Solo columnas solicitadas) ---
                            # Requerimiento: Solo n_disparo, estimacion, modelo. Sin request_id ni índice.
                            cols_mostrar = ['n_disparo', 'estimacion', 'modelo']
                            cols_finales = [c for c in cols_mostrar if c in df_filtrada_vista.columns]
                            st.dataframe(df_filtrada_vista[cols_finales], use_container_width=True, hide_index=True, height=200)

                            st.divider()
                            
                            # ========================================================
                            # NUEVA SECCIÓN: COMPARACIÓN CON SISMICIDAD REAL (CSV) + CALIBRACIÓN EXCEL
                            # ========================================================
                            st.subheader("Comparación: Real vs Predicho")
                            st.markdown("Para alinear correctamente la fecha con el número de disparo, se requieren ambos archivos:")

                            c1_load, c2_load = st.columns(2)
                            with c1_load:
                                file_real = st.file_uploader("1. Cargar Datos Sismicidad Reales (.csv)", type=['csv'])
                            with c2_load:
                                file_map = st.file_uploader("2. Cargar Excel Original (Mapeo N° Disparo <-> Fecha)", type=['xlsx'])

                            df_merged_real = None

                            if file_real and file_map:
                                st.info("Procesando y cruzando archivos...")
                                try:
                                    # A) PROCESAR SISMICIDAD (POR FECHA)
                                    df_sismica_raw = pd.read_csv(file_real)
                                    df_real_procesado, err_real = procesar_sismicidad_real(df_sismica_raw)
                                    
                                    # B) PROCESAR MAPEO (FECHA -> N_DISPARO)
                                    df_mapa, err_map = extraer_mapeo_fecha_disparo(file_map)
                                    
                                    if err_real:
                                        st.error(f"Error en CSV Sismicidad: {err_real}")
                                    elif err_map:
                                        st.error(f"Error en Excel Mapeo: {err_map}")
                                    else:
                                        # C) CRUZAR INFORMACIÓN (MERGE)
                                        # df_real_procesado tiene: fecha_grupo, Mw_EQ_Real
                                        # df_mapa tiene: fecha_obj, n_disparo_map
                                        
                                        df_merged_real = pd.merge(
                                            df_real_procesado, 
                                            df_mapa, 
                                            left_on='fecha_grupo', 
                                            right_on='fecha_obj', 
                                            how='inner'
                                        )
                                        
                                        if df_merged_real.empty:
                                            st.warning("El cruce de fechas entre el CSV y el Excel no generó coincidencias. Verifica los formatos de fecha.")
                                        else:
                                            st.success(f"Calibración exitosa: {len(df_merged_real)} puntos coincidentes alineados por N° Disparo.")
                                            
                                except Exception as e:
                                    st.error(f"Error general en la comparación: {e}")

                            # --- 4. PREPARACIÓN DE DATOS (GRÁFICO Y TABLA) ---
                            # Inicializamos con todo (por si no hay datos reales)
                            df_pred_plot = df_filtrada_vista.copy()
                            df_real_plot = None

                            # LOGICA DE FILTRO (INTERSECCION PARA QUE SOLO MUESTRE SI AMBOS EXISTEN)
                            if df_merged_real is not None and not df_merged_real.empty:
                                # 1. Obtener disparos que tienen dato REAL
                                disparos_reales = set(df_merged_real['n_disparo_map'].unique())
                                # 2. Obtener disparos que tienen PREDICCION (en los modelos seleccionados)
                                disparos_pred = set(df_filtrada_vista['n_disparo'].unique())
                                
                                # 3. Interseccion: Solo disparos que existen en AMBOS lados
                                disparos_comunes = disparos_reales.intersection(disparos_pred)
                                
                                # 4. Filtrar DataFrames para el gráfico Y LA TABLA
                                df_real_plot = df_merged_real[df_merged_real['n_disparo_map'].isin(disparos_comunes)].copy()
                                df_pred_plot = df_filtrada_vista[df_filtrada_vista['n_disparo'].isin(disparos_comunes)].copy()
                                
                                if len(disparos_comunes) == 0:
                                     st.warning("No hay coincidencias exactas de N° Disparo entre los datos reales y las predicciones.")

                            # --- 5. GRÁFICO (Usando data filtrada) ---
                            st.subheader("Gráfico de Resultados por Modelo vs Realidad")
                            
                            fig = go.Figure()

                            # Paleta de colores
                            colors = dict(zip(modelos_totales, ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']))

                            # A) TRAZAR MODELOS PREDICHOS (USANDO DATA FILTRADA)
                            for modelo in modelos_seleccionados:
                                df_modelo = df_pred_plot[df_pred_plot['modelo'] == modelo].sort_values(by='n_disparo')
                                
                                fig.add_trace(go.Scatter(
                                    x=df_modelo['n_disparo'],
                                    y=df_modelo['estimacion'],
                                    mode='markers',
                                    name=f"{modelo} (Pred)",
                                    marker=dict(size=6, opacity=0.8, color=colors.get(modelo, 'grey'))
                                ))

                            # B) TRAZAR REALIDAD (USANDO DATA FILTRADA)
                            if df_real_plot is not None and not df_real_plot.empty:
                                fig.add_trace(go.Scatter(
                                    x=df_real_plot['n_disparo_map'], 
                                    y=df_real_plot['Mw_EQ_Real'],
                                    mode='markers',
                                    name="REALIDAD (Calibrada)",
                                    marker=dict(size=8, color='black', symbol='x')
                                ))

                            # Añadir Líneas de Umbral Constantes (0.6, 1.2, 1.8)
                            thresholds_fix = [0.6, 1.2, 1.8]
                            colores_umbral_fix = ['orange', 'darkred', 'purple']
                            for i, t in enumerate(thresholds_fix):
                                fig.add_hline(y=t, line_dash="dash", line_color=colores_umbral_fix[i], annotation_text=f"Umbral {t}", annotation_position="top right")
                                    
                            fig.update_layout(
                                title="Estimación (Pred) vs Realidad (Calibrada por Fecha)",
                                xaxis_title="N° Disparo",
                                yaxis_title="Magnitud (Mw EQ)",
                                template="plotly_white",
                                height=550,
                                hovermode="x unified",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # --- EVALUACIÓN RMSE POR MODELO ---
                            if df_merged_real is not None and not df_merged_real.empty:
                                st.subheader("Evaluación de Modelos: RMSE vs Realidad")
                                
                                rmse_data = []
                                for modelo in modelos_seleccionados:
                                    # Filtrar predicciones del modelo (YA FILTRADAS ARRIBA)
                                    df_pred_modelo = df_pred_plot[df_pred_plot['modelo'] == modelo].copy()
                                    
                                    # Hacer merge con realidad por n_disparo (Usamos los DFs ya filtrados)
                                    df_compare = pd.merge(
                                        df_pred_modelo[['n_disparo', 'estimacion']], 
                                        df_real_plot[['n_disparo_map', 'Mw_EQ_Real']], 
                                        left_on='n_disparo', 
                                        right_on='n_disparo_map', 
                                        how='inner'
                                    )
                                    
                                    if not df_compare.empty:
                                        # Calcular RMSE
                                        rmse = np.sqrt(np.mean((df_compare['estimacion'] - df_compare['Mw_EQ_Real'])**2))
                                        
                                        # Clasificar calidad
                                        if rmse < 0.3:
                                            calidad = "Excelente"
                                        elif rmse < 0.5:
                                            calidad = "Bueno"
                                        elif rmse < 0.8:
                                            calidad = "Regular"
                                        else:
                                            calidad = "Malo"
                                        
                                        rmse_data.append({
                                            "Modelo": modelo,
                                            "RMSE": f"{rmse:.4f}",
                                            "Calidad": calidad,
                                            "N° Comparaciones": len(df_compare)
                                        })
                                    else:
                                        rmse_data.append({
                                            "Modelo": modelo,
                                            "RMSE": "N/A",
                                            "Calidad": "Sin datos coincidentes",
                                            "N° Comparaciones": 0
                                        })
                                
                                if rmse_data:
                                    df_rmse = pd.DataFrame(rmse_data)
                                    st.table(df_rmse)
                                    
                                    st.markdown("""
                                    **Interpretación RMSE:**
                                    - **Excelente** (< 0.3): Error muy bajo, predicciones altamente precisas
                                    - **Bueno** (0.3 - 0.5): Error aceptable, predicciones confiables
                                    - **Regular** (0.5 - 0.8): Error moderado, predicciones útiles con cautela
                                    - **Malo** (> 0.8): Error alto, predicciones poco confiables
                                    """)

                            # --- 5. TABLA RESUMEN (Usando data FILTRADA df_pred_plot) ---
                            st.subheader("Tabla Resumen de Umbrales por Modelo")
                            resumen_data = []

                            for modelo in modelos_seleccionados:
                                # AQUI ESTA EL CAMBIO: Usamos df_pred_plot en lugar de df_filtrada_vista
                                df_modelo_vals = df_pred_plot[df_pred_plot['modelo'] == modelo]['estimacion']
                                total_vals = len(df_modelo_vals)
                                
                                c_gt_0 = (df_modelo_vals > 0).sum()
                                c_gt_06 = (df_modelo_vals > 0.6).sum()
                                c_gt_12 = (df_modelo_vals > 1.2).sum()
                                c_gt_18 = (df_modelo_vals > 1.8).sum()
                                
                                resumen_data.append({
                                    "Modelo": modelo,
                                    "Total Valores": total_vals,
                                    "> 0": c_gt_0,
                                    "> 0.6": c_gt_06,
                                    "> 1.2": c_gt_12,
                                    "> 1.8": c_gt_18
                                })

                            # AGREGAR FILA DE REALIDAD SI EXISTE (Usando df_real_plot filtrado)
                            if df_real_plot is not None and not df_real_plot.empty:
                                real_vals = df_real_plot['Mw_EQ_Real']
                                total_real = len(real_vals)
                                
                                r_gt_0 = (real_vals > 0).sum()
                                r_gt_06 = (real_vals > 0.6).sum()
                                r_gt_12 = (real_vals > 1.2).sum()
                                r_gt_18 = (real_vals > 1.8).sum()
                                
                                resumen_data.append({
                                    "Modelo": "REALIDAD",
                                    "Total Valores": total_real,
                                    "> 0": r_gt_0,
                                    "> 0.6": r_gt_06,
                                    "> 1.2": r_gt_12,
                                    "> 1.8": r_gt_18
                                })
                                
                            if resumen_data:
                                df_resumen = pd.DataFrame(resumen_data)
                                st.table(df_resumen)

                            

                    else:
                        # Si faltan columnas clave, mostramos el dataframe completo por defecto
                        st.warning("El formato de datos recibido no contiene las columnas esperadas (n_disparo, estimacion, modelo). Se muestra tabla cruda.")
                        st.dataframe(df_res, use_container_width=True)

                else:
                    st.info("El estado es DONE, pero no se encontraron datos en request_result.")

# ============================================================
# 4. LOGICA DE CONTROL PRINCIPAL
# ============================================================

if not st.session_state['logged_in']:
    login_screen()
else: 
    main_app()
