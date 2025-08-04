---
permalink: /projets/
title: "Projets"
classes: wide
---

Sur cette page vous pouvez trouver les contenus sur lesquels j'ai travaillé (à titre personnel ou professionnel) mais qui sont référencés sur d'autres sites que mon blog personnel. Il s'agit principalement de traductions de cours, et des créations de jeux de données et de modèles.

<br><br>
# Vue d'ensemble


<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600&display=swap');

.timeline-container {
    max-width: 2000px;
    margin: 10px auto 10px;
    padding: 20px;
    overflow-x: scroll;
    font-family: 'DM Sans', sans-serif;
}

/* .timeline-container::-webkit-scrollbar {
    display: none;  /* Chrome, Safari, Edge 
} */

.timeline-container {
    scrollbar-width: auto;  
    scrollbar-color: #888 #f1f1f1;  
}

.timeline {
    position: relative;
    min-width: 1200px; /* Minimum width before scrolling */
}

.line {
    position: absolute;
    width: 100%;
    height: 2px;
    background: #333;
    bottom: 30px;
}

.timeline-items {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    position: relative;
    min-height: 150px;
}

.month-marker {
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
    flex: 1;
}

.month-dot {
    width: 12px;
    height: 12px;
    background: #000;
    border-radius: 50%;
    margin-bottom: -14px;
    position: relative;
    z-index: 1;
}

.month-label {
    font-weight: bold;
    margin-top: 10px;
}

.events-container {
    position: absolute;
    bottom: 40px;
    left: 50%;
    transform: translateX(-50%);
    width: 250px;
    text-align: left;
}

.event {
    position: relative;
    left: 92px;
    margin: 8px 0;
    font-size: 14px;
    display: flex;
    align-items: flex-start;
    gap: 5px;
    white-space: pre-wrap;
    max-width: 210px;
}

.event img {
    width: 20px;
    height: 20px;
    vertical-align: middle;
}

.event a {
    /*color: #000000;*/
    text-decoration: none;
}

.event a:hover {
    color: #686868;
    text-decoration: underline;
}
</style>

<!-- Based on https://huggingface.co/lvwerra/science-timeline-->
<div id="snake-timeline-container" style="max-width: 1000px; margin: 20px auto; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
   <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; position: relative;">
      <!-- First Row: Left to Right -->
      <div style="display: flex; margin-bottom: 32px; min-height: 140px; position: relative;">
         <div style="position: absolute; top: 8px; left: 0; right: 0; height: 1px; background: #e0e0e0; z-index: 1;"></div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 600; font-size: 14px; color: #1a1a1a; background: #f0f0f0; padding: 4px 12px; border-radius: 12px; margin-bottom: 16px; display: inline-block;">SEP 2023</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
            <a href="https://huggingface.co/collections/CATIE-AQ/catie-french-qa-pack-650821750f44c341cdb8ec91" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">❓ QAmemBERT<br>(110 or 336M parameters, 512 tokens)</div>
               </a>
              <a href="https://huggingface.co/collections/CATIE-AQ/catie-french-prompts-datasets-6508208ad55dd4e15cd67f8b" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">📝 DFP<br>(Dataset of French Prompts)<br>(+100M rows covering 30 different NLP tasks)</div>
               </a>
            </div>
         </div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">OCT</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
               <!-- Empty -->
            </div>
         </div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">NOV</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
               <!-- Empty -->
            </div>
         </div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">DEC</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
               <!-- Empty -->
            </div>
         </div>
      <div style="position: absolute; width: 1px; height: calc(100% + 32px); background: #e0e0e0; top: 8px; right: 0; z-index: 1;"></div> 
      </div>
      <!-- Second Row: Right to Left -->
      <div style="display: flex; margin-bottom: 32px; min-height: 140px; position: relative;">
         <div style="position: absolute; top: 8px; left: 0; right: 0; height: 1px; background: #e0e0e0; z-index: 1;"></div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-right: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">APR</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
              <a href="https://huggingface.co/collections/CATIE-AQ/catie-french-long-sequences-datasets-6513de50d06aa2cf17e5b64c" target="_blank" style="text-decoration: none; color: inherit;">
            <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">🐘 Long sequences (8K to 256K tokens) French QA and summarization datasets</div>
            </div>
         </div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-right: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">MAR</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
               <!-- Empty -->
            </div>
         </div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-right: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">FEB</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
            <a href="https://github.com/catie-aq/flashT5" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">⚡ FAT5 model v0<br>(Flash Attention T5)</div>
               </a>
            <a href="https://huggingface.co/datasets/CATIE-AQ/frenchSTS" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">👥 FrenchSTS<br>(Sentence Similarity dataset in French)</div>
               </a>
            </div>
            </div>
      <div style="display: flex; margin-bottom: 32px; min-height: 140px; position: relative;">
         <div style="position: absolute; top: 8px; left: 0; right: 0; height: 1px; background: #e0e0e0; z-index: 1;"></div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-right: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 600; font-size: 14px; color: #1a1a1a; background: #f0f0f0; padding: 4px 12px; border-radius: 12px; margin-bottom: 16px; display: inline-block;">JAN 2024</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
              <a href="https://huggingface.co/collections/CATIE-AQ/catie-french-ner-pack-658aefafe3f7a2dcf0e4dbb4" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">🔍 NERmemBERT<br>(110 or 336M parameters, 512 tokens, in 3 or 4 entities)</div>
               </a>
            </div>
            </div>
         </div>
      <div style="position: absolute; width: 1px; height: calc(100% + 32px); background: #e0e0e0; top: 8px; left: 0; z-index: 1;"></div>
      </div>
      </div>
      <!-- Third Row: Left to Right -->
      <div style="display: flex; margin-bottom: 32px; min-height: 140px; position: relative;">
         <div style="position: absolute; top: 8px; left: 0; right: 0; height: 1px; background: #e0e0e0; z-index: 1;"></div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">MAY</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
               <!-- Empty -->
            </div>
         </div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">JUN</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
               <!-- Empty -->
            </div>
         </div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">JUL</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
            <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">🔒 In the vault, hopefully one day I'll be able to get permission to put this online.</div>
            </div>
         </div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">AUG</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
               <!-- Empty -->
            </div>
         </div>
        <div style="position: absolute; width: 1px; height: calc(100% + 32px); background: #e0e0e0; top: 8px; right: 0; z-index: 1;"></div> 
      </div>
      <!-- Fourth Row: Right to Left -->
      <div style="display: flex; margin-bottom: 32px; min-height: 140px; position: relative;">
         <div style="position: absolute; top: 8px; left: 0; right: 0; height: 1px; background: #e0e0e0; z-index: 1;"></div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-right: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">DEC</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
               <!-- Empty -->
            </div>
         </div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-right: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">NOV</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
            <a href="https://huggingface.co/CATIE-AQ/QAmemberta" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">❓ QAmemBERTa<br>(111M parameters, 1024 tokens)</div>
               </a>
              <a href="https://huggingface.co/CATIE-AQ/NERmemberta-3entities" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">🔍 NERmemBERTa<br>(111M parameters, 1024 tokens)</div>
               </a>
            </div>
         </div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-right: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">OCT</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
               <!-- Empty -->
            </div>
         </div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-right: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">SEP</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
               <!-- Empty -->
            </div>
         </div>
      <div style="position: absolute; width: 1px; height: calc(100% + 32px); background: #e0e0e0; top: 8px; left: 0; z-index: 1;"></div> 
      </div>
      <!-- Five Row: Right to Left -->
      <div style="display: flex; margin-bottom: 32px; min-height: 140px; position: relative;">
         <div style="position: absolute; top: 8px; left: 0; right: 0; height: 1px; background: #e0e0e0; z-index: 1;"></div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 600; font-size: 14px; color: #1a1a1a; background: #f0f0f0; padding: 4px 12px; border-radius: 12px; margin-bottom: 16px; display: inline-block;">JAN 2025</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
            <a href="https://huggingface.co/spaces/CATIE-AQ/FAT5-report" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">⚡ FAT5 (official release)<br>(147M parameters trained from scratch for €1,200)</div>
               </a>
            </div>
         </div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">FEB</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
              <a href="https://huggingface.co/collections/CATIE-AQ/french-vqa-datasets-678a607a4c08258a5212950b" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">👁️ French VQA datasets</div>
               </a>
              <a href="https://huggingface.co/collections/CATIE-AQ/french-caption-datasets-678a612dc1223672ae4e157d" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">✍️ French caption datasets</div>
              <a href="https://huggingface.co/collections/CATIE-AQ/french-visual-retriever-datasets-678a6206c04f6a166139102c" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">🐶 French visual retriever datasets</div>
               </a>
            </div>
         </div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">MAR</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
            <a href="https://huggingface.co/spaces/CATIE-AQ/Guide_Evaluation_LLM" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">⚖️ French translation of the The LLM Evaluation guidebook by Clémentine Fourrier</div>
               </a>
            </div>
         </div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">APR</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
            <a href="https://huggingface.co/CATIE-AQ/ModernQAmembert" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">❓ ModernQAmembert<br>(136M parameters, 8192 tokens)</div>
               </a>
              <a href="https://huggingface.co/CATIE-AQ/Moderncamembert_3entities" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">🔍 modernNERmemBERT<br>(136M parameters, 8192 tokens, 3 or 4 entities)</div>
               </a>
            </div>
         </div>
      <div style="position: absolute; width: 1px; height: calc(100% + 32px); background: #e0e0e0; top: 8px; right: 0; z-index: 1;"></div> 
      </div>
      <!-- Six Row: Right to Left -->
      <div style="display: flex; margin-bottom: 32px; min-height: 140px; position: relative;">
         <div style="position: absolute; top: 8px; left: 0; right: 0; height: 1px; background: #e0e0e0; z-index: 1;"></div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-right: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">AUG</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
              <a href="https://huggingface.co/learn/agents-course/fr/unit0/introduction" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">French translation of<br>Hugging Face 🤗<br>AI Agents Course<br></div>
               </a>
            </div>
         </div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-right: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">JUL</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
              <a href="https://huggingface.co/collections/CATIE-AQ/catie-french-sparse-embedding-6888de67763acdc901a39833" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">📏 French sparse embedding V0<br></div>
               </a>
              <a href="https://huggingface.co/datasets/CATIE-AQ/frenchNLI" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">➡️ FrenchNLI<br></div>
               </a>
              <a href="https://huggingface.co/collections/CATIE-AQ/french-table-to-text-datasets-6888d96418fa7b3dc0a60557" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">𝄜 French table-to-text datasets<br></div>
               </a>
            </div>
         </div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-right: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">JUN</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
              <a href="https://huggingface.co/collections/CATIE-AQ/nanobeir-fr-6846d5b34c70acb3d4d5af52" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">🍺 NanoBEIR-fr<br>(for French IR evaluation)</div>
               </a>
              <a href="https://huggingface.co/collections/CATIE-AQ/xmrec-french-part-reviews-and-metadata-datasets-6888e4bb77b68acf6039c6f7" target="_blank" style="text-decoration: none; color: inherit;">
                  <div style="padding: 6px 8px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; line-height: 1.4; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0,0,0,0.05); cursor: pointer;" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.1)'; this.style.background='#e3f2fd'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 2px rgba(0,0,0,0.05)'; this.style.background='#f8f9fa'">🛒 XMrec datasets<br>(French part)</div>
               </a>
            </div>
         </div>
         <div style="flex: 1; padding: 0 15px; text-align: center; position: relative; z-index: 2;">
            <div style="width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-right: 10px solid #666; margin: 2px auto 12px;"></div>
            <div style="font-weight: 500; font-size: 13px; color: #666; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px;">MAY</div>
            <div style="font-size: 12px; color: #4a4a4a; max-width: 180px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px;">
               <!-- Empty -->
            </div>
         </div>
      </div>
   </div>
</div>


# Modèles et jeux de données en français

## FAT5
Le FAT5 est une implémentation du [T5](https://arxiv.org/abs/1910.10683) en PyTorch avec un objectif [UL2](https://arxiv.org/abs/2205.05131) optimisé pour GPGPU développé avec [Boris ALBAR](https://b-albar.github.io/portfolio/).  
Elle utilise des noyaux CUDA et Triton personnalisés ainsi que des optimisations spécifiques pour augmenter le débit et réduire l'utilisation de la mémoire pour l'entraînement et l'inférence d'un facteur 2 par rapport à l'implémentation originale disponible dans Hugging Face.  
Nous l'avons appliquée en pré-entraînant un modèle en français de 147M paramètres en utilisant uniquement une A100. Nous estimons ainsi pouvoir ramener le prix de pré-entraînement d'un tel modèle à seulement 1200€ (estimation faite sur une instance Sesterce).   
Le code de pré-entrainement est disponible sur [GitHub](https://github.com/catie-aq/flashT5) sous licence Apache-2.0 et les poids du modèle entraîné sur le compte [Hugging Face du CATIE]([https://huggingface.co/CATIE-AQ](https://huggingface.co/collections/CATIE-AQ/catie-french-fat5-ul2-677697a35feea336389d6403)). Un article de blog détaillant notre méthodologie est disponible [ici](https://huggingface.co/spaces/CATIE-AQ/FAT5-rapport).

## NER
Les NERmemBERT constituent une famille de modèles de Reconnaissance d’Entités Nommées en français capable d’étiqueter jusqu’à 4 entités (Personnalités, Lieux, Organisations, Divers tel que des noms d’œuvre, de maladies, etc.). Ils sont disponibles en taille base (110M ou 136M de paramètres) et large (336M), gérant des contextes allant de 512 à 8192 tokens. Les poids sont disponibles gratuitement en open-source, tout comme les jeux de données ayant servis à l’entraînement. Le tout est disponible sur le compte [Hugging Face du CATIE](https://huggingface.co/collections/CATIE-AQ/catie-french-ner-pack-658aefafe3f7a2dcf0e4dbb4). Un article de blog détaillant la méthodologie adoptée est disponible [ici](https://lbourdois.github.io/blog/NER/).<br>
Ils ont été téléchargés plus de 180 000 fois depuis leur mise en ligne.

## Question Answering
Les QAmemBERT constituent une famille de réponse aux questions en français capable d’indiquer si la réponse à une question est présente ou pas dans un texte de contexte associé. Ils sont disponibles en taille base (110M ou 136M de paramètres) et large (335M), gérant des contextes allant de 512 à 8192 tokens. Les poids sont disponibles gratuitement en open-source, tout comme le jeu de données ayant servis à l’entraînement. Le tout est disponible sur le compte [Hugging Face du CATIE](https://huggingface.co/collections/CATIE-AQ/catie-french-qa-pack-650821750f44c341cdb8ec91). Un article de blog détaillant la méthodologie adoptée est disponible [ici](https://lbourdois.github.io/blog/QA/).<br>
Ils ont été téléchargés plus de 155 000 fois depuis leur mise en ligne.

## DFP
*Dataset of French Prompts* (DFP) contient 113 129 978 lignes portant sur 30 tâches de NLP différentes.  
724 prompts ont été écrits sous forme impérative, de tutoiement et de vouvoiement afin de couvrir autant que possible les données de pré-entraînement utilisées par le modèle qui utilisera ces données et qui nous sont inconnues.

Les colonnes *inputs* et *targets* suivent le même format que l'ensemble de données [xP3](https://huggingface.co/datasets/bigscience/xP3) de Muennighoff et al.  
L'ensemble des détails est disponible sur [Hugging Face](https://huggingface.co/datasets/CATIE-AQ/DFP).<br>
Il a été téléchargé plus de 65 000 fois depuis sa mise en ligne.

## La marmite
Projet toujours en cours.  
Il s'agit de proposer un équivalent en français du jeu de données [*the cauldron*](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) afin de pouvoir entraîner un VLM en français.  
Ce jeu de données prendra en compte des données d'OCR (voir celles déjà disponibles en ligne [ici](https://huggingface.co/collections/lbourdois/french-ocr-datasets-67c8d3152330f11227e0d108)), de captionning (voir celles déjà disponibles en ligne [ici](https://huggingface.co/collections/lbourdois/french-caption-datasets-67c8d2227284a5daa00c50b9)), de VQA (voir celles déjà disponibles en ligne [ici](https://huggingface.co/collections/lbourdois/french-vqa-datasets-67c8d1a162a23ef0e9a2bc89)) et de raisonnement.  
Les sous-jeux de données déjà en ligne ont été téléchargé plus de 15 000 fois depuis leur mise en ligne.

<br><br><br>

# Traductions
## Cours de Yann Le Cun et Alfredo Canziani de la NYU
Cette traduction a été la plus longue à effectuer s'étalant de 2020 à 2022.  
Le contenu est structuré en 19 unités réparties sur 33 vidéos 🎥 de cours (cours magistraux et travaux dirigés) d’une durée totale d’environ 45H, 74 pages web 🌐 résumant les vidéos via les notes prises par les étudiants pendant le cours, et 16 notebooks Jupyter 📓 (en PyTorch) utilisés lors des TD. Enfin un jeu de données de plus de 3000 données parallèles vérifiées manuellement a été créé pour entraîner un modèle de traduction.  
Vous pouvez retrouver toutes ces ressources sur le site internet dédié qui a été conçu à l'occasion : [https://lbourdois.github.io/cours-dl-nyu/](https://lbourdois.github.io/cours-dl-nyu/).

## Cours de Hugging Face 🤗
### Cours de NLP
En 2022, j'ai traduit le cours de traitement automatique du langage de Hugging Face.  
Le contenu est structuré en 10 chapitres comprenant un total de 76 vidéos 🎥 d'une durée totale d'environ 5H, de 78 pages web 🌐 et 61 notebooks Jupyter 📓  (en PyTorch et Tensorflow).  
Vous pouvez retrouver toutes ces ressources sur le site de [Hugging Face](https://huggingface.co/learn/nlp-course/fr/chapter1/1).

### Cours d'audio
En 2023, j'ai traduit le cours de traitement automatique du langage de Hugging Face.  
Le contenu est structuré en 8 unités réparties sur 46 pages web 🌐.  
Vous pouvez retrouver toutes ces ressources sur le site de [Hugging Face](https://huggingface.co/learn/audio-course/fr/).

### Cours sur les modèles de diffusion
En 2023, j'ai traduit le cours sur les modèles de diffusion de Hugging Face.  
Le contenu est structuré en 4 chapitres portant sur 17 pages web 🌐 et 8 notebooks Jupyter 📓 (en PyTorch).  
Vous pouvez retrouver toutes ces ressources sur le GitHub de [Hugging Face](https://github.com/huggingface/diffusion-models-class/tree/main/units/fr) (le contenu n'ayant pas encore été propagé sur le site officiel).

### Cours sur les agents IA
En 2025, j'ai participé avec [Kim NOEL](https://github.com/knoel99) à la traduction du cours sur les agents IA de Hugging Face.  
Le contenu est structuré en 4 unités (+ 3 bonus) réparties sur 74 pages web 🌐 et 16 notebooks Jupyter 📓.  
Vous pouvez retrouver toutes ces ressources sur le site de [Hugging Face](https://huggingface.co/learn/agents-course/fr).
