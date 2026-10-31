import torch

def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model

def train_gan(model, data_loader, criterion=None, optimizer=None, device='cpu', epochs=10, lr=2e-4):
    """Train a GAN (model.generator + model.discriminator) with the standard adversarial loss.

    criterion: defaults to BCEWithLogitsLoss (the discriminator outputs logits).
    optimizer: (generator_optimizer, discriminator_optimizer); defaults to Adam(lr, betas=(0.5, 0.999)).
    Per-epoch average losses are stored in model.history.
    """
    gen, disc = model.generator, model.discriminator
    model.to(device)

    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()
    if optimizer is None:
        optimizer = (torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999)),
                     torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999)))
    opt_g, opt_d = optimizer

    model.history = {"d_loss": [], "g_loss": [], "d_real": [], "d_fake": []}
    for epoch in range(epochs):
        model.train()
        d_total = g_total = real_total = fake_total = 0.0

        for real, _ in data_loader:
            real = real.to(device)
            batch_size = real.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # === Train Discriminator: maximize log D(x) + log(1 - D(G(z))) ===
            noise = torch.randn(batch_size, model.z_dim, device=device)
            fake = gen(noise)
            d_real = disc(real)
            d_fake = disc(fake.detach())
            d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # === Train Generator: maximize log D(G(z)) (non-saturating loss) ===
            g_loss = criterion(disc(fake), real_labels)

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            d_total += d_loss.item()
            g_total += g_loss.item()
            real_total += torch.sigmoid(d_real).mean().item()
            fake_total += torch.sigmoid(d_fake).mean().item()

        n = len(data_loader)
        for k, v in zip(model.history, (d_total / n, g_total / n, real_total / n, fake_total / n)):
            model.history[k].append(v)
        print(f"Epoch {epoch+1}/{epochs} | D loss {d_total/n:.4f} | G loss {g_total/n:.4f} "
              f"| D(x) {real_total/n:.3f} | D(G(z)) {fake_total/n:.3f}")

    return model
