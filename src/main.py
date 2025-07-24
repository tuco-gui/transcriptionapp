import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, render_template, jsonify
import whisper
import tempfile

app = Flask(__name__, template_folder='../templates', static_folder='../static')

app.config['UPLOAD_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads'))

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = whisper.load_model("base")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo de áudio enviado'}), 400
    
    file = request.files['audio_file']
    
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400

    if file and (file.filename.endswith('.mp3') or file.filename.endswith('.ogg')):
        temp_path = None
        try:
            _, temp_ext = os.path.splitext(file.filename)
            temp_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=temp_ext, dir=app.config['UPLOAD_FOLDER'], prefix='upload_')
            temp_path = temp_file_obj.name
            file.save(temp_path)
            temp_file_obj.close()

            result = model.transcribe(temp_path, language='pt', word_timestamps=False) 
            
            full_transcription = result.get('text', '')
            
            return jsonify({'transcription': full_transcription.strip()})
        except Exception as e:
            print(f"Error during transcription: {str(e)}", file=sys.stderr)
            return jsonify({'error': 'Ocorreu um erro interno ao processar o áudio.'}), 500
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e_remove:
                    print(f"Error removing temp file {temp_path}: {str(e_remove)}", file=sys.stderr)
    else:
        return jsonify({'error': 'Formato de arquivo inválido. Use MP3 ou OGG.'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

