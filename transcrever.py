import datetime
import os
from faster_whisper import WhisperModel
from moviepy import VideoFileClip
from tqdm import tqdm
import torch
import sys 
import shutil

# --- CONFIGURAÇÕES ---
# Verifique se a GPU está disponível e configure o dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if torch.cuda.is_available() else "int8"

tamanho_modelo = "large-v3"
diretorio_videos = r"C:\Users\iebt\Videos"
diretorio_saida = r"C:\Users\iebt\Videos\Transcricao"
# -------------------

# Cria o diretório de saída se ele não existir
os.makedirs(diretorio_saida, exist_ok=True)

print("Carregando o modelo Whisper... (Isso pode levar alguns minutos na primeira vez)")
model = WhisperModel(tamanho_modelo, device=device, compute_type=compute_type)
print("✅ Modelo carregado com sucesso!")

# Lista todos os arquivos .mp4 no diretório de vídeos
arquivos_mp4 = [f for f in os.listdir(diretorio_videos) if f.endswith('.mp4')]

if not arquivos_mp4:
    print(f"\n🚫 Nenhum arquivo .mp4 encontrado no diretório: '{diretorio_videos}'.")
    print("Nenhuma transcrição foi realizada.")
    print("Processo finalizado.")
    sys.exit(0) # Termina o script aqui com sucesso
    
# Loop principal com a barra de progresso para ARQUIVOS
for arquivo_mp4 in tqdm(arquivos_mp4, desc="Progresso Geral (Arquivos)"):
    caminho_completo_video = os.path.join(diretorio_videos, arquivo_mp4)
    caminho_temporario_audio = os.path.join(diretorio_saida, "temp_audio.wav")

    # 1. Extrair áudio do vídeo
    try:
        # Redirecionar stdout para suprimir as mensagens do MoviePy
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w') # Abre o "buraco negro" para onde as mensagens irão
                
        video_clip = VideoFileClip(caminho_completo_video)
        total_duration_seconds = video_clip.audio.duration
        video_clip.audio.write_audiofile(caminho_temporario_audio, codec='pcm_s16le')
        video_clip.close()
        
        sys.stdout.close() # Fecha o "buraco negro"        
        sys.stdout = original_stdout # Restaurar stdout

        # Mensagem concisa após a extração
        print(f"    ✅ Áudio extraído de '{arquivo_mp4}'. Duração: {datetime.timedelta(seconds=int(total_duration_seconds))}")
                
    except Exception as e:
        print(f"\nErro ao extrair áudio de {arquivo_mp4}: {e}")
        continue

    total_duration_formatted = datetime.timedelta(seconds=int(total_duration_seconds))
    # 2. Transcrever o áudio
    # Consumir o gerador segments para que possamos iterar sobre ele várias vezes, se necessário,
    # ou para saber o tamanho total para uma barra de progresso interna se fosse usar tqdm.
    # No seu caso, basta passar direto ao loop.
    segments, info = model.transcribe(caminho_temporario_audio, beam_size=5,word_timestamps=True)
    
    print(f"    - Iniciando transcrição para: '{arquivo_mp4}'")
    print(f"        Idioma detectado: '{info.language}' com probabilidade {info.language_probability:.2f}. Duração total: {total_duration_formatted}")

    # Lista para armazenar os textos de cada segmento
    linhas_transcritas = []

    # É bom converter 'segments' para uma lista aqui para poder calcular o progresso com base no número total de segmentos,
    # ou para ter um iterável que não seja consumido de uma vez.
    # No entanto, o `model.transcribe` retorna um gerador que pode ser iterado diretamente.
    # Para o seu formato de progresso, iterar diretamente está OK.
    
    for segment in segments:
        # Tempo de progresso atual (usando o final do segmento para uma percepção mais próxima do "passou")
        progress_seconds = segment.end
        progress_formatted = datetime.timedelta(seconds=int(progress_seconds))

        # Cálculo da porcentagem
        progress_percentage = (progress_seconds / total_duration_seconds) * 100           

    # --- NOVA BARRA DE PROGRESSO INTERNA ---
    # Envolvemos o gerador 'segments' com tqdm para ver o progresso por segmento
    #for segment in tqdm(segments, desc="Progresso no Arquivo Atual (Segmentos)", unit=" seg"):

        # Monta a linha de progresso
        progress_line = (
            f"Progresso: {progress_formatted} / {total_duration_formatted} "
            f"({progress_percentage:.2f}%)"
        )

        # Monta a linha do segmento
        segment_line = f"Segmento: [{segment.start:.2f}s -> {segment.end:.2f}s]"
        
        # Obter o tamanho do terminal para limpar a linha completamente
        terminal_width = shutil.get_terminal_size().columns
        output_str = f"\r{progress_line} | {segment_line} "
        sys.stdout.write(output_str.ljust(terminal_width)) # Preenche com espaços até o final da linha
        sys.stdout.flush() # Força a exibição imediata
        
        linhas_transcritas.append(segment.text.strip())
    
    # Limpa completamente a linha de progresso após o loop de segmentos para este arquivo
    sys.stdout.write('\r' + ' ' * terminal_width + '\r')
    sys.stdout.flush()
    print(f"    ✅ Transcrição de '{arquivo_mp4}' concluída.")
    
    # 3. Salvar a transcrição em um arquivo de texto
    texto_completo = "\n".join(linhas_transcritas)
    nome_arquivo_saida = os.path.splitext(arquivo_mp4)[0] + ".txt"
    caminho_arquivo_saida = os.path.join(diretorio_saida, nome_arquivo_saida)

    with open(caminho_arquivo_saida, 'w', encoding='utf-8') as f:
        f.write(texto_completo)
    print(f"    💾 Transcrição salva em '{nome_arquivo_saida}'.") # Mensagem clara de salvamento

    # 4. Remover o arquivo de áudio temporário
    os.remove(caminho_temporario_audio)

print("\n🎉 Processo concluído! Todas as transcrições foram salvas.")