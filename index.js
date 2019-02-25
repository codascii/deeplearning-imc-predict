//  https://www.has-sante.fr/portail/upload/docs/application/pdf/2009-09/table_imc_230909.pdf

//  Le réseau
const reseau = tf.sequential();

//  Couche des entrées
/*const coucheEntrees = tf.layers.dense({
    units: 2,
    inputShape: [2],
    activation: 'sigmoid'
});
            reseau.add(coucheEntrees);*/

//  Couche cachée
const coucheCachee = tf.layers.dense({
    units: 4,
    inputShape: [2],
    activation: 'sigmoid'
});
reseau.add(coucheCachee);

//  Couche de sortie
const coucheSortie = tf.layers.dense({
    units: 3,
    activation: 'sigmoid'
});
reseau.add(coucheSortie);

const sgdOptimizer = tf.train.sgd(0.1);
reseau.compile({
    optimizer: sgdOptimizer,
    loss: tf.losses.meanSquaredError
});
            //  Données d'apprentissage
const lesEntreesDeTest = tf.tensor2d([[145, 50],
    [145, 55],
    [145, 60],
    [145, 65],
    [145, 70],
    [145, 75],
    [145, 80],
    [145, 85],
    [145, 90],
    [145, 95],
    [145, 100],
    [150, 50],
    [150, 55],
    [150, 60],
    [150, 65],
    [150, 70],
    [150, 75],
    [150, 80],
    [150, 85],
    [150, 90],
    [150, 95],
    [150, 100],
    [155, 50],
    [155, 55],
    [155, 60],
    [155, 65],
    [155, 70],
    [155, 75],
    [155, 80],
    [155, 85],
    [155, 90],
    [155, 95],
    [155, 100],
    [160, 50],
    [160, 55],
    [160, 60],
    [160, 65],
    [160, 70],
    [160, 75],
    [160, 80],
    [160, 85],
    [160, 90],
    [160, 95],
    [160, 100],
    [165, 50],
    [165, 55],
    [165, 60],
    [165, 65],
    [165, 70],
    [165, 75],
    [165, 80],
    [165, 85],
    [165, 90],
    [165, 95],
    [165, 100],
    [170, 50],
    [170, 55],
    [170, 60],
    [170, 65],
    [170, 70],
    [170, 75],
    [170, 80],
    [170, 85],
    [170, 90],
    [170, 95],
    [170, 100],
    [175, 50],
    [175, 55],
    [175, 60],
    [175, 65],
    [175, 70],
    [175, 75],
    [175, 80],
    [175, 85],
    [175, 90],
    [175, 95],
    [175, 100],
    [180, 50],
    [180, 55],
    [180, 60],
    [180, 65],
    [180, 70],
    [180, 75],
    [180, 80],
    [180, 85],
    [180, 90],
    [180, 95],
    [180, 100],
    [185, 50],
    [185, 55],
    [185, 60],
    [185, 65],
    [185, 70],
    [185, 75],
    [185, 80],
    [185, 85],
    [185, 90],
    [185, 95],
    [185, 100],
    [190, 50],
    [190, 55],
    [190, 60],
    [190, 65],
    [190, 70],
    [190, 75],
    [190, 80],
    [190, 85],
    [190, 90],
    [190, 95],
    [190, 100],
    [195, 50],
    [195, 55],
    [195, 60],
    [195, 65],
    [195, 70],
    [195, 75],
    [195, 80],
    [195, 85],
    [195, 90],
    [195, 95],
    [195, 100],
    [200, 50],
    [200, 55],
    [200, 60],
    [200, 65],
    [200, 70],
    [200, 75],
    [200, 80],
    [200, 85],
    [200, 90],
    [200, 95],
    [200, 100],
    [205, 50],
    [205, 55],
    [205, 60],
    [205, 65],
    [205, 70],
    [205, 75],
    [205, 80],
    [205, 85],
    [205, 90],
    [205, 95],
    [205, 100],
    [210, 50],
    [210, 55],
    [210, 60],
    [210, 65],
    [210, 70],
    [210, 75],
    [210, 80],
    [210, 85],
    [210, 90],
    [210, 95],
    [210, 100],
    [145, 110],
    [145, 120],
    [145, 130],
    [145, 140],
    [145, 150],
    [145, 160],
    [150, 110],
    [150, 120],
    [150, 130],
    [150, 140],
    [150, 150],
    [150, 160],
    [155, 110],
    [155, 120],
    [155, 130],
    [155, 140],
    [155, 150],
    [155, 160],
    [160, 110],
    [160, 120],
    [160, 130],
    [160, 140],
    [160, 150],
    [160, 160],
    [165, 110],
    [165, 120],
    [165, 130],
    [165, 140],
    [165, 150],
    [165, 160],
    [170, 110],
    [170, 120],
    [170, 130],
    [170, 140],
    [170, 150],
    [170, 160],
    [175, 110],
    [175, 120],
    [175, 130],
    [175, 140],
    [175, 150],
    [175, 160],
    [180, 110],
    [180, 120],
    [180, 130],
    [180, 140],
    [180, 150],
    [180, 160],
    [185, 110],
    [185, 120],
    [185, 130],
    [185, 140],
    [185, 150],
    [185, 160],
    [190, 110],
    [190, 120],
    [190, 130],
    [190, 140],
    [190, 150],
    [190, 160],
    [195, 110],
    [195, 120],
    [195, 130],
    [195, 140],
    [195, 150],
    [195, 160],
    [200, 110],
    [200, 120],
    [200, 130],
    [200, 140],
    [200, 150],
    [200, 160],
    [205, 110],
    [205, 120],
    [205, 130],
    [205, 140],
    [205, 150],
    [205, 160],
    [210, 110],
    [210, 120],
    [210, 130],
    [210, 140],
    [210, 150],
    [210, 160]
]);

const lesSortiesDeTest = tf.tensor2d([[0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],

    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],    
    
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
        
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9], 
    
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9], 
    
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],

    [0.9, 0.1, 0.1],
    
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],

    [0.1, 0.1, 0.9],
    
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],

    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],

    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    [0.9, 0.1, 0.1],
    
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],

    [0.1, 0.9, 0.1],
    
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],

    [0.1, 0.9, 0.1],

    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],

    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],

    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],

    [0.1, 0.9, 0.1],
    [0.1, 0.9, 0.1],
    
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.1, 0.9]
]);

/*reseau.predict(tf.tensor2d([
    [171, 71],
    [101, 271],
    [131, 121],
    [11, 91],
])).print();*/
async function apprendre() {
    let spanInfoTauxErreur = document.getElementById('taux_erreur');
    for (var i = 1; i < 1000; i++) {
        const config = {
            shuffle: true
        };

        //await reseau.fit(lesEntreesDeTest, lesSortiesDeTest, config);
        const resultat = await reseau.fit(lesEntreesDeTest, lesSortiesDeTest, config);
        spanInfoTauxErreur.innerHTML = resultat.history.loss[0];
        //console.log(resultat.history.loss[0]);
    }
}

            /*const entrees = tf.tensor2d(
                [[150, 98]]
            );*/

            /*apprendre().then(() => {
                console.log("Apprentissage terminé !");
                let sorties = reseau.predict(entrees);
                sorties.print();
            });*/

            const gen = () => {
                // Première partie : affichage des inputs
                for (let t = 145; t <= 210; t += 5) {
                    for (let p = 50; p <= 100; p += 5) {
                        console.log("[" + t +", " + p + "],");
                    }

                    for (let p = 110; p <= 160; p += 10) {
                        console.log("[" + t +", " + p + "],");
                    } 
                }

                console.log("##########################################################################");

                // Première partie : affichage des outputs
                for (let t = 145; t <= 210; t += 5) {
                    for (let p = 50; p <= 100; p += 5) {
                        let imc = Math.ceil(p / Math.pow(t/100, 2));

                        if (imc <= 18)
                            console.log("[1, 0, 0],");
                        else if(imc <= 29) 
                            console.log("[0, 1, 0],");
                        else
                            console.log("[0, 0, 1],");
                    }

                    for (let p = 110; p <= 160; p += 10) {
                        let imc = Math.ceil(p / Math.pow(t/100, 2));

                        if (imc <= 18)
                            console.log("[1, 0, 0],");
                        else if(imc <= 29) 
                            console.log("[0, 1, 0],");
                        else
                            console.log("[0, 0, 1],");
                    }
                }
            }

            const gen2 = () => {
                // Première partie : affichage des inputs
                for (let t = 145; t <= 210; t += 5) {
                    for (let p = 110; p <= 160; p += 10) {
                        console.log("[" + t +", " + p + "],");
                    }                    
                }

                // Première partie : affichage des outputs
                for (let t = 145; t <= 210; t += 5) {
                    for (let p = 110; p <= 160; p += 10) {
                        let imc = Math.ceil(p / Math.pow(t/100, 2));

                        if (imc <= 18)
                            console.log("[1, 0, 0],");
                        else if(imc <= 29) 
                            console.log("[0, 1, 0],");
                        else
                            console.log("[0, 0, 1],");
                    }
                }
            }

//  Évènement du click sur le button "Lancer l'apprentissage"
const lancerLapprentissage = () => {
    let info = document.getElementById('info');
    info.innerHTML = "Apprentissage en cours...";
    info.classList.remove('text-info');
    info.classList.add('text-primary');

    document.getElementById('runLearning').disabled = true;
    document.getElementById('runLearning').textContent = "Apprentissage en cours...";

    apprendre().then(() => {
        info.innerHTML = "Apprentissage terminé, vous pouvez commencer !";
        info.classList.remove('text-primary');
        info.classList.add('text-success');

        document.getElementById('taille').disabled = false;
        document.getElementById('poids').disabled = false;
        document.getElementById('btn-predire').disabled = false;
        
        document.getElementById('runLearning').disabled = false;
        document.getElementById('runLearning').textContent = "Relancer l'apprentissage";
    });
}
var d = 0;

async function predire() {
    const taille = parseInt(document.getElementById('taille').value);
    const poids = parseInt(document.getElementById('poids').value);
    let resultatMince, resultatNormal, resultatSurPoids;
    console.log("Dans prédir : taille " + taille);
    console.log("Dans prédir : poids " + poids);
    
    let sorties = reseau.predict(tf.tensor2d([[taille, poids]]));
    sorties.print();
    //  Résultats
    d = await sorties.data();

    resultatMince = (d[0] * 100).toPrecision(4);
    resultatNormal = (d[1] * 100).toPrecision(4);
    resultatSurPoids = (d[2] * 100).toPrecision(4);

    //  Enlève le resultat sélectionner
    var selected = document.querySelector('.silhouette-container.preded');
    if (selected) selected.classList.remove('preded');

    if (resultatMince > resultatNormal && resultatMince > resultatSurPoids) {        
        // Sélectionner mince
        document.getElementById('mince').classList.add('preded');
    } else if (resultatNormal > resultatMince && resultatNormal > resultatSurPoids) {        
        // Sélectionner normal
        document.getElementById('normal').classList.add('preded');
    } else {        
        // Sélectionner surpoids
        document.getElementById('surpoids').classList.add('preded');
    }

    document.getElementById('pred_mince').innerHTML = resultatMince + " %";
    document.getElementById('pred_normal').innerHTML = resultatNormal + " %";
    document.getElementById('pred_surpoids').innerHTML = resultatSurPoids + " %";
}



